# Exercise 4

For this exercise we were tasked on implementing a following behaviour on our Duckiebot where it will follow a leader bot if it detects one around the Duckietown environment. This involves combining different components from previous exercises such as lane following, Apriltag detection, and custom LED emitter patterns. With new components like vehicle detection and distance calculation. All of these components are fed into a single node that handles the robot and lane following behaviour.

## Running the demo

First start the lane following pipeline from Ducketown by running

```bash
./run_script.sh
```

Then you can run the program by running this at the repo's root:

```bash
dts devel build -f -H $BOT && dts devel run -H $BOT
```

If pulling, and building the images are taking too long you can build and run the container locally by running this instead:

```bash
dts devel build -f && dts devel run -R $BOT
```