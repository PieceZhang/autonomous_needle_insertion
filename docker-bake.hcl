// docker-bake.hcl — Build all images in one pass, sharing the base stage.
// Usage:  docker buildx bake              (all targets)
//         docker buildx bake app ndi      (subset)
//         docker buildx bake --no-cache   (force rebuild)

variable "UBUNTU_MIRROR" { default = "" }
variable "ROS2_MIRROR"   { default = "" }

group "default" {
  targets = ["app", "ndi", "franka", "console"]
}

target "app" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "app"
  tags       = ["aniros-app:jazzy"]
  args = {
    UBUNTU_MIRROR = UBUNTU_MIRROR
    ROS2_MIRROR   = ROS2_MIRROR
  }
}

target "ndi" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "ndi"
  tags       = ["aniros-ndi:jazzy"]
  args = {
    UBUNTU_MIRROR = UBUNTU_MIRROR
    ROS2_MIRROR   = ROS2_MIRROR
  }
}

target "franka" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "franka"
  tags       = ["aniros-franka:jazzy"]
  args = {
    UBUNTU_MIRROR = UBUNTU_MIRROR
    ROS2_MIRROR   = ROS2_MIRROR
  }
}

target "console" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "console"
  tags       = ["aniros-guidance-console:latest"]
  args = {
    UBUNTU_MIRROR = UBUNTU_MIRROR
    ROS2_MIRROR   = ROS2_MIRROR
  }
}
