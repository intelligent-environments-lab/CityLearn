# CityLearn
CityLearn is an open source OpenAI Gym environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. A major challenge for RL in demand response is the ability to compare algorithm performance. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/dr.jpg)

## Environment Overview

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters.

![Citylearn]()
<object data="https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/citylearn_systems.pdf" type="application/pdf">
    <embed src="https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/citylearn_systems.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/citylearn_systems.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Installation
Install latest release in PyPi with `pip`:
```console
pip install CityLearn
```

## Documentation
Refer to the [docs](https://intelligent-environments-lab.github.io/CityLearn/) for documentation of the CityLearn API.