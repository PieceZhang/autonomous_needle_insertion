#!/usr/bin/env bash
set -euo pipefail

# 从所有 /dev/video* 里，按编号从小到大遍历
HAGIBIS_DEV=""

for dev in /dev/video*; do
  # 读出该设备在内核里的名字，如 "Hagibis UHCapture"
  name_file="/sys/class/video4linux/$(basename "$dev")/name"
  if [[ -r "$name_file" ]]; then
    name=$(cat "$name_file")
  else
    continue
  fi

  # 名字里包含 "Hagibis" 就认为是我们要的设备
  if [[ "$name" == *Hagibis* ]]; then
    HAGIBIS_DEV="$dev"
    break   # 因为 for 是按 /dev/video0,1,2... 顺序遍历，所以第一个就是编号最小的
  fi
done

if [[ -z "$HAGIBIS_DEV" ]]; then
  echo "USB_grabber 'Hagibis' not found" >&2
  exit 1
fi

echo "Selected USB_grabber 'Hagibis' is: $HAGIBIS_DEV"
echo "Set device ID in .env file"

export USB_VIDEO_DEVICE="$HAGIBIS_DEV"

#docker compose --profile dev up -d usb_video_grabber