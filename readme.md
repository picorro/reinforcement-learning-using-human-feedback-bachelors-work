Python ver 3.8.10 </br>

Installing requirements: </br>
```pip install -r requirements.txt``` </br>
Baseline 1 usage example: </br>
```python main.py -e "LunarLander-v2" -a "dqn"  -s 10000000 -g True -m "baseline1" -nm "<run name>"``` </br>
</br>
Recording a gameplay dataset: </br>
```python main.py -m "play" -nm "example-dataset" -e "LunarLander-v2"``` </br>
</br>
Baseline 2 usage example: </br>
```python main.py -e "LunarLander-v2" -a "dqn"  -s 10000000 -g True  -m "baseline2" -d "<relative dataset path>" -nm "<run name>"``` </br>
</br>
Baseline 3 usage example: </br>
```python main.py -e "LunarLander-v2" -a "dqn"  -s 10000000 -g True  -m "baseline3" -d "<relative dataset path>" -nm "<run name>" -interventions 3``` </br>
</br>
Tensorboard logs are generated in tensorboard_logs folder, have in mind that baseline3 automatically combines the logs only if it has finished training. Otherwise you will need to combine the logs with the tensorboard_combiner.py file manually.
