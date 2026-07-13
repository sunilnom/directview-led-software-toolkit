/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2026 Intel Corporation
 */

/*
 * mtl_tx.c — Intel Media Transport Library (MTL) pipeline TX helpers.
 *
 * Provides format-mapping, frame-copy, session lifecycle, and frame-send
 * functions used exclusively by the direct MTL TX path (ENABLE_MTL_TX).
 * None of this file is compiled in the default FFmpeg-avdevice build.
 *
 * Functions:
 *   get_transport_format()    — AVPixelFormat → st20_fmt (wire format)
 *   get_input_format()        — AVPixelFormat → st_frame_fmt (buffer layout)
 *   get_st_fps()              — integer fps   → st_fps (MTL enum)
 *   mtl_copy_crop_to_frame()  — crop AVFrame luma/chroma into st_frame addr[]
 *   mtl_tx_init()             — initialise the MTL library instance
 *   mtl_tx_uninit()           — release the MTL library instance
 *   mtl_tx_session_create()   — create one ST20P TX session
 *   mtl_tx_session_free()     — destroy one ST20P TX session
 *   mtl_tx_send_yuv_frame()   — get MTL frame, copy crop, put frame
 *   mtl_tx_send_raw_yuv()     — get MTL frame, memcpy raw buffer, put frame
 */

/* mtl_tx_init / mtl_tx_uninit are compiled unconditionally so that the
 * FFmpeg avdevice path can also pre-initialise DPDK EAL with ALL NIC ports
 * before opening any mtl_st20p session (fixes multi-NIC support).
 * All other functions remain guarded by ENABLE_MTL_TX. */
#include "mtl/mtl_tx.h"
#include "app_context.h"
#include "core/session_manager.h"
#include "util/logger.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <time.h>

#ifdef ENABLE_MTL_TX
#include <libavutil/pixdesc.h>

/* =========================================================================
 * get_input_format / get_transport_format
 * =========================================================================
 *
 * get_input_format:     AVPixelFormat → st_frame_fmt  (in-memory buffer layout)
 * get_transport_format: AVPixelFormat → st20_fmt      (on-wire ST 2110-20 packing)
 *
 * These are DIFFERENT enums in MTL. Passing a st_frame_fmt value into
 * transport_fmt causes the numeric value to be reinterpreted as a different
 * wire format (e.g. ST_FRAME_FMT_YUV444PLANAR10LE=8 → ST20_FMT_RGB_8BIT=8).
 */
enum st_frame_fmt get_input_format(enum AVPixelFormat fmt) {
  switch (fmt) {
    case AV_PIX_FMT_YUV422P10LE: return ST_FRAME_FMT_YUV422PLANAR10LE;
    case AV_PIX_FMT_YUV420P:     return ST_FRAME_FMT_YUV420CUSTOM8;
    case AV_PIX_FMT_YUV444P10LE: return ST_FRAME_FMT_YUV444PLANAR10LE;
    case AV_PIX_FMT_GBRP10LE:    return ST_FRAME_FMT_GBRPLANAR10LE;
    case AV_PIX_FMT_YUV422P12LE: return ST_FRAME_FMT_YUV422PLANAR12LE;
    case AV_PIX_FMT_YUV444P12LE: return ST_FRAME_FMT_YUV444PLANAR12LE;
    case AV_PIX_FMT_GBRP12LE:    return ST_FRAME_FMT_GBRPLANAR12LE;
    default:
      LOG_ERROR("get_input_format: unsupported AVPixelFormat %d", fmt);
      return (enum st_frame_fmt)-1;
  }
}

enum st20_fmt get_transport_format(enum AVPixelFormat fmt) {
  switch (fmt) {
    case AV_PIX_FMT_YUV422P10LE: return ST20_FMT_YUV_422_10BIT;
    case AV_PIX_FMT_YUV420P:     return ST20_FMT_YUV_420_8BIT;
    case AV_PIX_FMT_YUV444P10LE: return ST20_FMT_YUV_444_10BIT;
    case AV_PIX_FMT_GBRP10LE:    return ST20_FMT_RGB_10BIT;
    case AV_PIX_FMT_YUV422P12LE: return ST20_FMT_YUV_422_12BIT;
    case AV_PIX_FMT_YUV444P12LE: return ST20_FMT_YUV_444_12BIT;
    case AV_PIX_FMT_GBRP12LE:    return ST20_FMT_RGB_12BIT;
    default:
      LOG_ERROR("get_transport_format: unsupported AVPixelFormat %d", fmt);
      return (enum st20_fmt)-1;
  }
}

/* =========================================================================
 * get_st_fps
 * =========================================================================
 *
 * Map an integer frames-per-second value to the MTL st_fps enum.
 * MTL requires an exact fps enum for its transmission schedule; arbitrary
 * fractional rates are not supported by the current ST 2110-21 scheduler.
 */
enum st_fps get_st_fps(int fps) {
  switch (fps) {
    case 25: return ST_FPS_P25;
    case 30: return ST_FPS_P30;
    case 50: return ST_FPS_P50;
    case 60: return ST_FPS_P60;
    default:
      LOG_WARN("get_st_fps: unsupported fps %d, defaulting to ST_FPS_P30", fps);
      return ST_FPS_P30;
  }
}

/* =========================================================================
 * mtl_copy_crop_to_frame
 * =========================================================================
 *
 * Copy a rectangular crop of a planar FFmpeg AVFrame into an MTL st_frame.
 *
 * dst->addr[0/1/2] point to MTL's DMA-mapped hugepage TX buffers (luma,
 * Cb, Cr planes respectively).  MTL expects tightly-packed rows with no
 * per-line stride padding (stride == crop_w * bps for luma).
 *
 * The AVPixFmtDescriptor is consulted for:
 *   - bps:          bytes per sample (supports 8-bit and 10LE packed).
 *   - log2_chroma_w: horizontal chroma subsampling shift (0 = 4:4:x, 1 = 4:2:x).
 *   - log2_chroma_h: vertical   chroma subsampling shift (0 = no subsamp, 1 = 4:2:0).
 *
 *   fmt          — pixel format of src (must match desc used to alloc MTL session)
 *   crop_x/y     — top-left corner in the full-width shared yuv_frame (luma coords)
 *   crop_w/h     — crop rectangle dimensions in luma pixels
 */
void mtl_copy_crop_to_frame(struct st_frame* dst, const AVFrame* src,
                             int crop_x, int crop_y,
                             int crop_w, int crop_h,
                             enum AVPixelFormat fmt) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
  if (!desc || !dst || !src) return;

  /* E-4: Bounds check — prevent out-of-bounds read from src */
  if (crop_x < 0 || crop_y < 0 || crop_w <= 0 || crop_h <= 0) {
    LOG_ERROR("mtl_copy_crop_to_frame: invalid crop rect (x=%d y=%d w=%d h=%d)",
              crop_x, crop_y, crop_w, crop_h);
    return;
  }
  if (crop_x + crop_w > src->width || crop_y + crop_h > src->height) {
    LOG_ERROR("mtl_copy_crop_to_frame: crop rect (x=%d+w=%d=%d, y=%d+h=%d=%d) "
              "exceeds source (%dx%d)",
              crop_x, crop_w, crop_x + crop_w,
              crop_y, crop_h, crop_y + crop_h,
              src->width, src->height);
    return;
  }

  int bps          = (desc->comp[0].depth + 7) / 8; /* bytes per sample         */
  int chroma_w_shl = desc->log2_chroma_w;            /* 0=4:4:x  1=4:2:x         */
  int chroma_h_shl = desc->log2_chroma_h;            /* 0=no-v-sub  1=4:2:0      */
  int chroma_h     = crop_h >> chroma_h_shl;
  int chroma_y     = crop_y >> chroma_h_shl;
  int chroma_x     = crop_x >> chroma_w_shl;
  int chroma_w     = crop_w >> chroma_w_shl;
  int dst_y_stride = crop_w  * bps;
  int dst_c_stride = chroma_w * bps;

  /* Resolve destination plane pointers.
   * For CUSTOM8 formats (e.g. YUV420CUSTOM8), MTL provides a single
   * contiguous buffer in addr[0] with addr[1]/addr[2] == NULL.  The
   * Y/U/V planes are packed sequentially: Y then U then V, each with
   * tightly-packed rows (no line padding).  Compute the offsets manually. */
  uint8_t* dst_y = (uint8_t*)dst->addr[0];
  if (!dst_y) return;

  uint8_t* dst_u = dst->addr[1] ? (uint8_t*)dst->addr[1]
                                 : dst_y + dst_y_stride * crop_h;
  uint8_t* dst_v = dst->addr[2] ? (uint8_t*)dst->addr[2]
                                 : dst_u + dst_c_stride * chroma_h;

  /* Luma (Y) plane — full crop_h rows, crop_w samples wide */
  for (int line = 0; line < crop_h; line++)
    memcpy(dst_y + line * dst_y_stride,
           src->data[0] + (crop_y + line) * src->linesize[0] + crop_x * bps,
           dst_y_stride);

  /* Cb (U) chroma plane — chroma_h rows, chroma_w samples wide */
  if (src->data[1]) {
    for (int line = 0; line < chroma_h; line++)
      memcpy(dst_u + line * dst_c_stride,
             src->data[1] + (chroma_y + line) * src->linesize[1] + chroma_x * bps,
             dst_c_stride);
  }

  /* Cr (V) chroma plane — chroma_h rows, chroma_w samples wide */
  if (src->data[2]) {
    for (int line = 0; line < chroma_h; line++)
      memcpy(dst_v + line * dst_c_stride,
             src->data[2] + (chroma_y + line) * src->linesize[2] + chroma_x * bps,
             dst_c_stride);
  }
}

#endif /* ENABLE_MTL_TX — mtl_copy_crop_to_frame and format helpers */

/* =========================================================================
 * PTP lock auto-detection and software-timestamp fallback
 * =========================================================================
 *
 * When built-in PTP is enabled, MTL runs its PTP client but does not, by
 * itself, switch the application to a software time source if no grandmaster
 * is present.  These helpers implement explicit auto-detect/fallback instead
 * of relying on MTL's implicit behaviour:
 *
 *   dvledtx_ptp_sync_notify() — registered as mtl_init_params.ptp_sync_notify.
 *     MTL calls it on every valid PTP DELAY_RESP from the grandmaster, so we
 *     use it to know PTP is (still) locked and record the last sync instant.
 *
 *   dvledtx_ptp_time_fn() — registered as mtl_init_params.ptp_get_time_fn and
 *     becomes MTL's time source.  It returns the hardware-disciplined PTP time
 *     while PTP is locked and automatically falls back to the system clock
 *     (software timestamp) if PTP never locks or stops syncing.
 *
 * This only applies to the direct MTL path: mtl_tx_init() owns mtl_init(),
 * whereas the FFmpeg mtl_st20p muxer creates its own MTL handle and thus keeps
 * MTL's implicit timing behaviour. */

#define DVLEDTX_NS_PER_SEC          1000000000ULL
/* Treat PTP as "locked" only if a sync arrived within this window. */
#define DVLEDTX_PTP_LOSS_TIMEOUT_NS (3ULL * DVLEDTX_NS_PER_SEC)
/* How long mtl_tx_log_ptp_status() waits for the first lock before reporting. */
#define DVLEDTX_PTP_LOCK_WAIT_MS    3000
#define DVLEDTX_PTP_POLL_MS         100

struct dvledtx_ptp_state {
  _Atomic unsigned long sync_count;     /* total PTP syncs received */
  _Atomic uint64_t      last_sync_mono; /* CLOCK_MONOTONIC ns of last sync */
  _Atomic int64_t       last_delta;     /* last reported phc delta (ns) */
  mtl_handle            mtl;            /* handle for mtl_ptp_read_time() */
};

static struct dvledtx_ptp_state g_ptp_state;

static uint64_t dvledtx_mono_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * DVLEDTX_NS_PER_SEC + (uint64_t)ts.tv_nsec;
}

static uint64_t dvledtx_realtime_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_sec * DVLEDTX_NS_PER_SEC + (uint64_t)ts.tv_nsec;
}

static bool dvledtx_ptp_is_locked(void) {
  if (atomic_load(&g_ptp_state.sync_count) == 0) return false;
  uint64_t last = atomic_load(&g_ptp_state.last_sync_mono);
  return (dvledtx_mono_ns() - last) < DVLEDTX_PTP_LOSS_TIMEOUT_NS;
}

static void dvledtx_ptp_sync_notify(void* priv, struct mtl_ptp_sync_notify_meta* meta) {
  (void)priv;
  if (meta == NULL) return;
  unsigned long prev = atomic_fetch_add(&g_ptp_state.sync_count, 1);
  atomic_store(&g_ptp_state.last_sync_mono, dvledtx_mono_ns());
  atomic_store(&g_ptp_state.last_delta, meta->delta);
  if (prev == 0)
    LOG_INFO("PTP: grandmaster detected (delta %ld ns) — using hardware PTP timestamps",
             (long)meta->delta);
}

/* MTL time source: hardware PTP while locked, system clock otherwise. */
static uint64_t dvledtx_ptp_time_fn(void* priv) {
  (void)priv;
  if (dvledtx_ptp_is_locked() && g_ptp_state.mtl != NULL)
    return mtl_ptp_read_time(g_ptp_state.mtl);
  return dvledtx_realtime_ns();
}

/* Intel Foxville (igc) device IDs — I225/I226 family. The DPDK e1000 PMD's
 * eth_igc_timesync_enable() segfaults on these parts, so MTL hardware PTP
 * (MTL_FLAG_PTP_ENABLE) cannot be used and we must fall back to software
 * timestamps BEFORE mtl_init() is called (the crash happens inside it). */
static const uint16_t k_igc_no_hw_ptp_ids[] = {
  0x15F2, /* I225-LM */        0x15F3, /* I225-V */   0x15F8, /* I225-I */
  0x15FD, /* I225 blank NVM */ 0x0D9F, /* I225-IT */
  0x3100, /* I225-K */         0x3101, /* I225-K2 */
  0x5502, /* I225-LMVP */      0x5504, /* I226-K */
  0x125B, /* I226-LM */        0x125C, /* I226-V */   0x125D, /* I226-IT */
  0x125E, /* I221-V */         0x125F, /* I226 blank NVM */
};

/* Read a hex integer from a PCI sysfs attribute (e.g. "device", "vendor"). */
static long dvledtx_read_pci_hex(const char* bdf, const char* attr) {
  char path[256];
  snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", bdf, attr);
  FILE* f = fopen(path, "re");
  if (f == NULL) return -1;
  long val = -1;
  if (fscanf(f, "%li", &val) != 1) val = -1; /* flawfinder: ignore */
  fclose(f);
  return val;
}

/* Return true if the NIC at this PCI BDF can safely enable MTL hardware PTP.
 * Known-broken parts (Intel igc/I225/I226) return false so the caller can
 * transparently downgrade to software timestamps instead of crashing. */
bool mtl_tx_nic_hw_ptp_supported(const char* bdf) {
  long vendor = dvledtx_read_pci_hex(bdf, "vendor");
  long device = dvledtx_read_pci_hex(bdf, "device");
  if (vendor < 0 || device < 0) {
    /* BDF of a DPDK-bound device is normally readable; if not, don't block. */
    LOG_WARN("PTP: cannot read PCI id for %s — proceeding with hardware PTP", bdf);
    return true;
  }
  if (vendor == 0x8086) {
    for (size_t i = 0; i < sizeof(k_igc_no_hw_ptp_ids) / sizeof(k_igc_no_hw_ptp_ids[0]); i++) {
      if ((uint16_t)device == k_igc_no_hw_ptp_ids[i]) {
        LOG_WARN("PTP: NIC %s (Intel igc 0x%04lx) DPDK PMD lacks working hardware "
                 "timesync — will use software (system-clock) timestamps", bdf, device);
        return false;
      }
    }
  }
  return true;
}

/* =========================================================================
 * mtl_tx_init / mtl_tx_uninit — compiled unconditionally (FFmpeg + direct)
 * =========================================================================
 *
 * mtl_tx_init() — initialise the MTL library using parameters from app.
 *   Stores the resulting handle in manager->mtl.
 *   Allocates one TX queue per session plus two spare queues, and minimal
 *   RX queues for MTL control traffic.
 *
 * mtl_tx_uninit() — release the MTL library instance.
 */
int mtl_tx_init(session_manager_t* manager, struct dvledtx_context* app) {
  struct mtl_init_params mtl_params;
  memset(&mtl_params, 0, sizeof(mtl_params));

  mtl_params.flags     = MTL_FLAG_BIND_NUMA | MTL_FLAG_DEV_AUTO_START_STOP;

  /* Hardware PTP capability guard: MTL_FLAG_PTP_ENABLE makes mtl_init() enable
   * NIC hardware timesync. Some DPDK PMDs (notably Intel igc / I225-I226) crash
   * in eth_*_timesync_enable() during that step, so detect unsupported NICs up
   * front and downgrade to software timestamps instead of segfaulting. The MTL
   * instance shares one global PTP setting, so a single unsupported port forces
   * software mode for all ports. */
  if (app->ptp_enable) {
    for (int ni = 0; ni < app->nic_count; ni++) {
      if (!mtl_tx_nic_hw_ptp_supported(app->nics[ni].port)) {
        app->ptp_enable = false;
        break;
      }
    }
    if (!app->ptp_enable)
      LOG_WARN("PTP: hardware PTP disabled (unsupported NIC present) — running "
               "in software (system-clock) timestamp mode");
  }

  /* PTP hardware sync: enable MTL's built-in PTP so the direct MTL pipeline
   * paces transmission against the NIC hardware clock (PHC) instead of TSC.
   * A custom time source (dvledtx_ptp_time_fn) is registered so the pipeline
   * auto-detects whether a grandmaster is present and transparently falls back
   * to software (system-clock) timestamps when PTP is not locked. Mirrors the
   * ptp_enable handling in the FFmpeg avdevice path (src/ffmpeg/ffmpeg_tx.c). */
  if (app->ptp_enable) {
    mtl_params.flags |= MTL_FLAG_PTP_ENABLE;
    atomic_store(&g_ptp_state.sync_count, 0UL);
    atomic_store(&g_ptp_state.last_sync_mono, 0ULL);
    atomic_store(&g_ptp_state.last_delta, (int64_t)0);
    g_ptp_state.mtl = NULL;
    mtl_params.ptp_sync_notify = dvledtx_ptp_sync_notify;
    mtl_params.ptp_get_time_fn = dvledtx_ptp_time_fn;
    LOG_INFO("MTL init: built-in PTP enabled with auto-detect and software fallback");
  }

  mtl_params.num_ports = app->nic_count;

  /* Count sessions assigned to each NIC for queue allocation */
  int* sessions_per_nic = calloc((size_t)app->nic_count, sizeof(int));
  if (sessions_per_nic == NULL) {
    LOG_ERROR("Failed to allocate sessions_per_nic array");
    return -1;
  }
  for (int i = 0; i < app->st20p_sessions; i++) {
    int ni = app->session_net[i].nic_index;
    if (ni >= 0 && ni < app->nic_count)
      sessions_per_nic[ni]++;
  }

  for (int ni = 0; ni < app->nic_count; ni++) {
    snprintf(mtl_params.port[ni], MTL_PORT_MAX_LEN, "%s", app->nics[ni].port);
    memcpy(mtl_params.sip_addr[ni], app->nics[ni].sip_addr, MTL_IP_ADDR_LEN);
    mtl_params.pmd[ni] = mtl_pmd_by_port_name(app->nics[ni].port);

    uint16_t tx_queues = (uint16_t)(sessions_per_nic[ni] + 2);
    uint16_t rx_queues = 1; /* minimal RX for control traffic; MTL adds 1 system queue */
    mtl_params.tx_queues_cnt[ni] = tx_queues;
    mtl_params.rx_queues_cnt[ni] = rx_queues;

    LOG_INFO("MTL init: port[%d]=%s pmd=%d tx_queues=%d rx_queues=%d",
             ni, app->nics[ni].port, mtl_params.pmd[ni], tx_queues, rx_queues);
  }

  free(sessions_per_nic);

  manager->mtl = mtl_init(&mtl_params);
  if (!manager->mtl) {
    LOG_ERROR("Failed to initialise MTL library");
    return -1;
  }
  /* Publish the handle for the adaptive PTP time source (used only when PTP
   * is enabled; harmless otherwise). Safe to set here because no session
   * transmits — and thus ptp_get_time_fn is not called — until later. */
  g_ptp_state.mtl = manager->mtl;
  LOG_INFO("MTL library initialised successfully (%d port(s))", app->nic_count);
  return 0;
}

void mtl_tx_uninit(session_manager_t* manager) {
  if (manager->mtl) {
    mtl_uninit(manager->mtl);
    manager->mtl = NULL;
  }
  g_ptp_state.mtl = NULL;
}

/* =========================================================================
 * mtl_tx_log_ptp_status
 * =========================================================================
 *
 * Report the active timing mode for the direct MTL path. Called after the MTL
 * device has started (first session created) so the built-in PTP client has
 * begun exchanging messages. Waits up to DVLEDTX_PTP_LOCK_WAIT_MS for a lock,
 * then logs whether hardware PTP or the software fallback is in effect.
 */
void mtl_tx_log_ptp_status(struct dvledtx_context* app) {
  if (!app->ptp_enable) {
    LOG_INFO("PTP: disabled by config — using software (system-clock) timestamps");
    return;
  }

  int waited = 0;
  while (waited < DVLEDTX_PTP_LOCK_WAIT_MS && !dvledtx_ptp_is_locked()) {
    usleep((useconds_t)DVLEDTX_PTP_POLL_MS * 1000);
    waited += DVLEDTX_PTP_POLL_MS;
  }

  if (dvledtx_ptp_is_locked())
    LOG_INFO("PTP: locked to grandmaster (delta %ld ns) — hardware timestamp mode active",
             (long)atomic_load(&g_ptp_state.last_delta));
  else
    LOG_WARN("PTP: no grandmaster found within %d ms — running in software "
             "(system-clock) timestamp mode", DVLEDTX_PTP_LOCK_WAIT_MS);
}

/* =========================================================================
 * Everything below requires the direct MTL TX path (ENABLE_MTL_TX).
 * ========================================================================= */
#ifdef ENABLE_MTL_TX

/* =========================================================================
 * mtl_tx_session_create / mtl_tx_session_free
 * =========================================================================
 *
 * mtl_tx_session_create() — build st20p_tx_ops from ctx/app and call
 *   st20p_tx_create().  Sets ctx->handle and ctx->frame_size on success.
 *
 * mtl_tx_session_free() — call st20p_tx_free() and clear ctx->handle.
 */
int mtl_tx_session_create(session_manager_t* manager, struct st20p_tx_ctx* ctx,
                           struct dvledtx_context* app, int session_idx) {
  struct st20p_tx_ops ops;
  memset(&ops, 0, sizeof(ops));

  ops.name = ctx->session_name;
  ops.priv = ctx;

  ops.port.num_port = 1;
  int nic = app->session_net[session_idx].nic_index;
  memcpy(ops.port.dip_addr[MTL_SESSION_PORT_P], app->nics[nic].dip_addr, MTL_IP_ADDR_LEN);
  snprintf(ops.port.port[MTL_SESSION_PORT_P], MTL_PORT_MAX_LEN, "%s", app->nics[nic].port);

  int udp_port = app->session_net[session_idx].udp_port;
  if (udp_port == 0) udp_port = (int)app->udp_port + (session_idx * 2);
  ops.port.udp_port[MTL_SESSION_PORT_P] = (uint16_t)udp_port;

  int payload_type = app->session_net[session_idx].payload_type;
  if (payload_type == 0) payload_type = app->payload_type;
  ops.port.payload_type = (uint8_t)payload_type;

  ops.width         = (uint32_t)ctx->crop_width;
  ops.height        = (uint32_t)ctx->crop_height;
  ops.fps           = get_st_fps(app->fps);
  ops.transport_fmt = get_transport_format(app->fmt);
  ops.input_fmt     = get_input_format(app->fmt);
  if ((int)ops.transport_fmt == -1 || (int)ops.input_fmt == -1) {
    LOG_ERROR("Unsupported pixel format %d for MTL ST20P TX session %d",
              app->fmt, session_idx);
    return -1;
  }
  ops.device        = ST_PLUGIN_DEVICE_AUTO;
  ops.framebuff_cnt = 3;
  ops.flags         = ST20P_TX_FLAG_BLOCK_GET;

  ctx->handle = st20p_tx_create(manager->mtl, &ops);
  if (!ctx->handle) {
    LOG_ERROR("Failed to create MTL ST20P TX session %d", session_idx);
    return -1;
  }
  ctx->frame_size = st20p_tx_frame_size(ctx->handle);
  LOG_INFO("ST20P TX session %d: MTL handle created, frame_size=%zu, "
           "crop=%dx%d+%d+%d udp_port=%d",
           session_idx, ctx->frame_size,
           ctx->crop_width, ctx->crop_height,
           ctx->crop_x_offset, ctx->crop_y_offset,
           udp_port);
  return 0;
}

void mtl_tx_session_free(struct st20p_tx_ctx* ctx) {
  if (ctx->handle) {
    st20p_tx_free(ctx->handle);
    ctx->handle = NULL;
  }
}

/* =========================================================================
 * mtl_tx_send_yuv_frame
 * =========================================================================
 *
 * Obtains a free DMA TX ring buffer from MTL, copies the crop strip from
 * src into it via mtl_copy_crop_to_frame(), sets the RTP timestamp, and
 * puts the frame back to MTL for transmission.
 *
 * Returns 0 on success, -1 on error (e.g. get_frame returned NULL after
 * timeout, or copy failed).
 */
int mtl_tx_send_yuv_frame(struct st20p_tx_ctx* ctx, const AVFrame* src,
                          int crop_x, int crop_y, int crop_w, int crop_h) {
  if (!ctx->handle || !src) return -1;

  struct st_frame* frame = st20p_tx_get_frame(ctx->handle);
  if (!frame) {
    LOG_ERROR("ST20P TX(%d): st20p_tx_get_frame returned NULL", ctx->idx);
    return -1;
  }

  mtl_copy_crop_to_frame(frame, src, crop_x, crop_y, crop_w, crop_h,
                         ctx->app->fmt);

  /* Use the shared frame counter for RTP timestamp so ALL sessions stamp the
   * same video frame identically — prevents inter-session clock drift.
   * Per-session frames_sent diverges under thread scheduling jitter.
   * frame_counter is incremented by the decode thread after all TX threads
   * have consumed the frame (post barrier_copied), so it is stable here. */
  uint32_t frame_num = ctx->shared_dec
                       ? (uint32_t)atomic_load(&ctx->shared_dec->frame_counter)
                       : ctx->frames_sent;
  frame->tfmt      = ST10_TIMESTAMP_FMT_MEDIA_CLK;
  frame->timestamp = frame_num * 90000 / (uint32_t)ctx->app->fps;

  int ret = st20p_tx_put_frame(ctx->handle, frame);
  if (ret < 0) {
    LOG_ERROR("ST20P TX(%d): st20p_tx_put_frame failed (ret=%d)", ctx->idx, ret);
    return -1;
  }

  ctx->frames_sent++;
  if (ctx->frames_sent % 100 == 0)
    LOG_INFO("ST20P TX(%d): sent %d frames (MTL)", ctx->idx, ctx->frames_sent);
  return 0;
}

/* =========================================================================
 * mtl_tx_send_raw_yuv
 * =========================================================================
 *
 * Obtains a free DMA TX ring buffer, copies frame_size bytes from
 * ctx->source_buffer (wrapping for loop playback) directly into MTL's
 * DMA buffer, then puts the frame for transmission.
 *
 * Returns 0 on success, -1 on error.
 */
int mtl_tx_send_raw_yuv(struct st20p_tx_ctx* ctx) {
  if (!ctx->handle || !ctx->source_buffer || ctx->source_size == 0) return -1;

  struct st_frame* frame = st20p_tx_get_frame(ctx->handle);
  if (!frame) {
    LOG_ERROR("ST20P TX(%d): st20p_tx_get_frame returned NULL", ctx->idx);
    return -1;
  }

  size_t frame_bytes = ctx->frame_size;
  if (ctx->current_pos + frame_bytes > ctx->source_size)
    ctx->current_pos = 0;

  if (frame->addr[0])
    memcpy(frame->addr[0], ctx->source_buffer + ctx->current_pos, frame_bytes);
  ctx->current_pos += frame_bytes;

  /* raw YUV path is always single-session (no shared_dec); per-session
   * frames_sent is the correct counter here. */
  frame->tfmt      = ST10_TIMESTAMP_FMT_MEDIA_CLK;
  frame->timestamp = ctx->frames_sent * 90000 / (uint32_t)ctx->app->fps;

  int ret = st20p_tx_put_frame(ctx->handle, frame);
  if (ret < 0) {
    LOG_ERROR("ST20P TX(%d): st20p_tx_put_frame (raw yuv) failed (ret=%d)",
              ctx->idx, ret);
    return -1;
  }

  ctx->frames_sent++;
  if (ctx->frames_sent % 100 == 0)
    LOG_INFO("ST20P TX(%d): sent %d frames (MTL raw yuv)", ctx->idx, ctx->frames_sent);
  return 0;
}

#endif /* ENABLE_MTL_TX */
