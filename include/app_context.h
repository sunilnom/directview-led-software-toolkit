/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2026 Intel Corporation
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <libavutil/pixfmt.h>

#define MAX_TX_SESSIONS  (8)
#define MAX_INTERFACES   (8)

/* Per-session network and crop parameters (populated from JSON tx_sessions[]) */
struct tx_session_net {
  uint16_t udp_port;
  uint8_t  payload_type;
  int      crop_x;
  int      crop_y;
  int      crop_w;
  int      crop_h;
  int      nic_index;   /* which NIC this session uses (index into port[]/sip_addr[]/dip_addr[]) */
};

/* Application context for TX sessions */
struct dvledtx_context {
  /* NIC configuration — arrays indexed [0..nic_count-1] */
  int     nic_count;                                       /* number of active NICs */
  char    port[MAX_INTERFACES][64];                        /* DPDK NIC PCI BDF per NIC */
  char    sip_addr_str[MAX_INTERFACES][INET_ADDRSTRLEN];   /* source IP string per NIC */
  uint8_t sip_addr[MAX_INTERFACES][4];                     /* source IP binary per NIC */
  char    dip_addr_str[MAX_INTERFACES][INET_ADDRSTRLEN];   /* destination IP string per NIC */
  uint8_t dip_addr[MAX_INTERFACES][4];                     /* destination IP binary per NIC */

  char tx_url[256];
  char config_file[256];
  uint16_t udp_port;
  uint8_t  payload_type;  /* RTP dynamic payload type (default: 96) */

  /* Video parameters */
  uint32_t width;
  uint32_t height;
  uint32_t scale_width;       /* output width after scaling (0 = no scaling) */
  uint32_t scale_height;      /* output height after scaling (0 = no scaling) */
  int fps;                    /* frames per second: 25, 30, 50, 60 */
  enum AVPixelFormat fmt;     /* e.g. AV_PIX_FMT_YUV422P10LE */

  /* Session controls */
  int st20p_sessions;
  bool exit;
  bool force_dhcp;
  int test_time_s;

  /* Per-session network + crop config (from JSON tx_sessions[]) */
  struct tx_session_net session_net[MAX_TX_SESSIONS];

  /* Optional log file path from config (empty = console logging only) */
  char log_file[256];

};
