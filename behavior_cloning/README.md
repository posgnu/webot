# Behavior Cloning agents

## Setup
* data directory is set to be a `../data`
* data format in `data.gzip`
  * list of (action, state)
  * state: int[row][col][rgb]
  * action: dict {"type": "mousedown", "x": 74, "y": 134, "timing": 1}
    * type = {"mousedown" | "mouseup"}
    * x = 0 ~ 159
    * y = 0 ~ 209
    * timing = 

## Run
Train the agent
```shellscript
$ make train
```