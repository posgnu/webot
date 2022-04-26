# webot

## Tasks
[Task list](https://docs.google.com/spreadsheets/d/1Z0VB4Cl-ysxnREl2bAoqCKAjDDPv6efOaOfGZjN_1TQ/edit?usp=sharing)

## Directory structure
* `data`: this directory stores a series of state and action pairs as a gzip
  * data format in `{env_name}_bot_data.gzip`
    * list of (state, action)
    * state: dict{"uttereance": str, "img": int[row x col][3]}
      * utterance: str
      * img: int[row x col][rgb]
    * action: dict {"type": str, "x": int, "y": int, "timing": 1}
      * type = {"mousedown" | "mouseup" | "click"}
      * x = 0 ~ 159
      * y = 0 ~ 209
      * timing = 
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

Add webdriver path to environment variables. webdriver should be located in the root directory.
```shellscript
$ ./setup.sh
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
 
## TODO
* When recording the expert demonstration, make the permission request required only once. For now, permission request is sent every episode.
* Figure out what is the flipping operation before prediction of CNN
* Quatize cursor coordinate in the action space
* chnage name to wobgym