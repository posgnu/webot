# Behavior Cloning agents

## Setup
* data directory is set to be a `../data`

## State and Action definition
state = (image: [160][160][3], utterance: string)
* action = (action_type, x_coordinate, y_coordinate)
  * action_type = "mousedown" | "mouseup" | "click"
  * x_coordinate, y_coordinate


## Run
Train the agent
```shellscript
$ make train
```