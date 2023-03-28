# Exercise 5

For this exercise we were tasked on implementing a machine learning model that will recongize numbers attached to the AprilTags in the Duckietown environment and stop once all 10 numbers have been correctly identified. The robot will drive autonomusly in Duckietown until all 10 numbers are found.

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
