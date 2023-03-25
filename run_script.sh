#!/bin/bash

echo "Running exercise 5"

dts duckiebot demo --demo_name lane_following --duckiebot_name $BOT --package_name duckietown_demos --image duckietown/dt-core:daffy-arm64v8
