# webot

## Directory structure
* `data`: this directory stores a series of state and action pairs as a gzip
* `output`: this directory also stores a series of state and action but it additionally includes other metadata for showing it on the web.
* `behavior_cloning`: Implements the behavior cloning agent
* `computergym`: gym environment for computer control task
* `record.py`: recording server script that can record expert demonstrations on MiniWob++
* `viewer`: web page source code for viewing demonstration data in `output` directory

## Setup
Setup Python environment
```shellscript
$ pipenv install
````

Open shell in the virtualenv
```shellscript
$ pipenv shell
```

## Demonstration format
```
demo = {
    taskName: string,
    utterance: string,
    reward: int,
    rawReward: int, // 1 if succeeded and -1 otherwise
    states: list[state] // each state is recorded whenever there is an action
}
```

```
state = {
    time,
    action,
    image: list[list[(R, G, B)]], // RGB 2D array
    dom
}
```


## Record human demonstrations
* Start recording server
```shell script
$ mkdir out
$ ./record.py out/ data/
```
* Open local miniWoB environment with `record` get paramter. You should use miniwob environment in `computergym` directory.
```
file:///path/to/computergym/miniwob/miniwob_interface/html/miniwob/click-test.html?record=true
```
* Open the viewer to check recorded demonstrations
```
file:///path/to/viewer/viewer.html
```
 