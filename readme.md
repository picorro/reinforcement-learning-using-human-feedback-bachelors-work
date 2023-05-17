Python ver 3.8.10 </br>

Installing requirements: </br>
```pip install -r requirements.txt``` </br>
Recording a gameplay dataset: </br>
```python main.py -m "play" -nm "example-dataset" -e "LunarLander-v2"``` </br>
Baseline 3 usage example: </br>
```python main.py -e "LunarLander-v2" -a "dqn"  -s 10000000 -g True  -m "baseline3" -d "<relative dataset path>" -nm "<run name>" -interventions 3``` </br>
