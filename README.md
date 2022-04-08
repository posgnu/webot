# webot

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
$ ./record.py out
```
* Open local miniWoB environment with `record` get paramter
```
file:///path/to/computergym/miniwob/miniwob_interface/html/miniwob/click-test.html?record=true
```
* Open the viewer to check recorded demonstrations
```
file:///path/to/viewer/viewer.html
```
 