# Multi-agent Collaborative Rehabilitation Outpatient Assistance System - *Rehab-Flow*
## Introduction
This project is developed using the CrewAI framework and implements a multi-agent
collaborative system to assist rehabilitation outpatient care. Multiple intelligent
agents coordinate to handle patient inquiry, medical record analysis, rehabilitation
recommendation, and clinical decision support, enabling seamless human-AI interaction
in the diagnostic workflow.

### Key Features:
- Multi-Agent Collaboration: Separate agents for inquiry, auxiliary check suggestions,
  rehabilitation plan generation, etc.
- Flow-Driven Architecture: CrewAI Flow manages task sequences with dynamic branching
  and conditional logic.
- Safety Mechanisms: Maximum iteration limits in loops prevent infinite processing cycles.
- Extensible & Modular: Easily add new agents or workflow nodes to support future
  multi-department applications.

### Use Case:
- Rehabilitation outpatient support
- Patient inquiry and information collection
- Personalized rehabilitation recommendation
- Clinical decision assistance and workflow optimization
  
## Preparation
To start the running of rehabilitation assistance system, you need to configurate the virtual environment first. 
```bash
cd ./rehab_flow_1
crewai install # the .venv will be install
source .venv/bin/activate
```

## Run the system
Simply run one command and start to gather patient information right away.
```bash
crewai flow kickoff
```

---

## Structure of system
### Crews
Crews are combined of multiple agents (sometimes only one agent form a crew)./
In our system, there are 4 crews in used, 1 crew is defined but not in used, instead we decide
to use a direct LLM call to complete the ***inquiry crew***'s task.

1. inquery_crew: Supplement and enrich the patient's infomation for the diagnostic agent to analyze. 
2. diagnosis_crew: Analysis the diagnosis possiblities, give suggested sypotom inquiry and recommand supplementary examinations for further confirmation.
3. possibilities_elimination_crew: To further eliminate diagnosis possibilities by offering suggested diagnostic dialectics, more auxilary checks and updating diagnosis result.
4. treatement_plan_crew: Create a customized treatment plan for patient with detailed excution direction.
5. multiagents_report_crew: Summarized the whole process of multi-agent work flow and their intermediate output, and review the diagnosis result.
   
Each of the above crews is combined of some relative Agents and their respective Tasks (one agent : one task),
In the same crew, agents will communicate and work together to achieve the final goals,\
they can work sequentially[https://docs.crewai.com/en/learn/sequential-process] or hierarchically [https://docs.crewai.com/en/learn/hierarchical-process].
- If they work sequantially, there are two mode: 
  1. Execute step-by-step by your instruction with multiple tasks.
     Need to set **Context** on a task to get output from other tasks. (to enable agent communication)
  2. Collaborative single task, nominate one lead agent, but can delegate to others.
     Need to set **allow_delegation=True** in all agents of the crew to enable task assignment and asking questions to other agents.
- If they work hierarchically, there is only one mode:
  Manager-led task dominate the crew, **define and nominate** a manager agent and Manager Agent will delegate to specialists,\
  Specialists Agent focus on their expertise, Manager coordinates everything.

You can define how agents and tasks work in ***xxx_crew.py***.
more detail about how to create a **crew**: [https://docs.crewai.com/en/concepts/collaboration]

In our project:
All the crews above are sequentially executed, step-by-step by your instruction with multiple tasks, no manager task.

#### Agents
The smallest excutive unit in multi-agent cooperation system.
Each agent defined a particular role, goal, backstory and which llm we shall use - you can think of it as role-play in AI prompt engineering : )

#### Tasks
What exactly an agent is responsible for. To define a tasks, you have to prepared a agent first. 
A task is combined of 3 parts:
- description
What exaclty will the task is about, have to coherent with the role of the agent.
- expected_output
Define how you would like the llm present the output, JSON, Markdown，String, List...
You better design a specific structure for you output to avoid trouble of invalid format causing failed parsing.
- agent
Which agent will execute this task (only one). Make sure to use the customized agent for this task.


### Flow
A complete flow is combined of multiple crews, each crew's output is stored as an intermediate variable in ***State*** -  a pre-defined class in CrewAI library.
Crews are executed in a linear order passing information/output/update through ***State***, no come back and forth trick here: )

In our flow - ***RehabFlow***, crews are execute based on *start()* and *listen()* decorator.

```python
class RehabFlow(Flow[PatientState]):
    """"Rehabilitation Inquiries, Diagnosis and Treatment Flow"""
    
    @start()
    def get_user_input(self):
        ...... # omit

    @listen(get_user_input)
    def inquiry_assistance(self, p_state):
        ...... # omit
```
Obviously, a pre-defined class ***State*** can't satisfy our requirement of holding and passing massive amount of information during the multi-aganet diagnosis.
Therefore, allow me to introduce to you the most important concept of our "Multi-agent Collaborative Rehabilitation Outpatient Assistance System": **PatientState**
```python
# Define our flow state
class PatientState(BaseModel):
    initial_inquiry: InitialInquriy = Field(default_factory=InitialInquriy)
    audience_level: str = ""
    process_outline: ProcessOutline = Field(default_factory=ProcessOutline)
    diagnosis_result: Dict[str, str] = {}
    treatment_plan: Dict[str, str] = {}
```
In  **PatientState**, initial_inquiry will record the initial patient information gathering from *get_user_input()*
- The initial patient information is quite a lot, we store them in another class for convenience: **InitialInquriy**
  
 ```python
# Define the structure of the intial inquiry
class InitialInquriy(BaseModel):
    main_complain: str = Field(default="", description="Main complaint of the patient")
    history_of_present_illness: str = Field(default="", description="History of present illness")
    past_medical_history: str = Field(default="", description="Past medical history")
    allergy_history: str = Field(default="", description="Allergy history")
    family_history: str = Field(default="", description="Family history")
    physical_examination: str = Field(default="", description="Physical examination findings")
    personal_history: str = Field(default="", description="Personal history of the patient")
    auxiliary_examination: Dict[str, str] = Field(default_factory=dict, description="Dictionary of auxiliary examinations conducted before the inquiry")
```
Attention: no matter it is in **PatientState** or **InitialInquriy**, you have to be careful to treat their class variable's **type** and **structure**.
Of course, you are welcome to modify them based on your need, then you need to change every place where we invoke these class variables (assign value, update, append).


- *audience_level* defined our target user's professional level, if a patient wants to use it, we should definitely make the intrpretation easy to understand.

- *process_outline* has a heavy duty as well, first it keep track of which crews were complished - *complete_sections* (crutial variable for loop termination condition check),
  it also record all the additional inquiries and answers in *supplementary_inquiries*, and the additional diagnostic dialectics happened in **diagnosis_crew** are stored in *suggested_diagnostic_dialectics*, as well as the suggested auxiliary checks and results which stored in *supplementary_auxiliary_examinations*. 
  These extra outline will help agent acurrately confirm the diagnosis.

```python
# Define the structure of our diagnosis process outline
class ProcessOutline(BaseModel):
    complete_sections: List[str] = Field(default_factory=list, description="List of completed sections")
    supplementary_inquiries: Dict[str,str] = Field(default_factory=dict, description="Dictionary of supplementary inquiries made during the process")
    suggested_diagnostic_dialectics: List[Dict[str, str]] = Field(default_factory=list, description="List of suggested diagnostic dialectics used in the process") 
    supplementary_auxiliary_examinations: Dict[str, str] = Field(default_factory=dict, description="Dictionary of auxiliary examinations conducted during the process")
```

---
## Potential Issues
### By the time of demo complete 
Even though this project has been tested and is up and running. 
There are still some concern to be mentioned by initial developer:
- All the test case is fake, we never actually ask a patient to tryout the "Multi-agent Collaborative Rehabilitation Outpatient Assistance System" and ask
  for their feeling of interaction logic and practical experience. Nether ask a professional medical worker to go through the process and give some practical advice.
- The simulating conversation of patient is based on genrating content of *ChatGTP.4o* since the author never went to a rehabilitation outpatient before, don't know what feels like to broke a arm or something.. Hopefully I will never know; ）
- In the actual practice of Rehabilitation Outpatient, a simple "break arm during skiing" problem may not involve MRI examination, a CT or X-ray will be sufficeint.
But our current dispatch logic is as accurate as possible, this may cause a over-kill suggestion/reaction in many places to pursing a excessice confirmation before get into the next section.
- **We need to find the balance between "overkill" and "underdo"**. In current stage, since the fine-tuned LLM and RAG Model are not integated in *Agent* （currently only using open source LLM ), I am try to find the balance point by consistently modify the Task Prompt, honestly, it did not work stable in open source LLM.
The most bold and innovative part of our system is to hand over the determination of whether the initial inquiry was thorough and whether the diagnosis was complete to LLM instead of a hardcode termination.

However, it act sometimes super strict and asked you more than 3 round of questions, even ask you repeated questions you just answered.
Sometimes it act super loose on judgement, once you feed into a bit information, it will take the insufficient information and make a "story", luckly we have multiple agents have the loop-over mechanism, which to a large degree, minimized the negative impact of the lazy judgement behavior, it will eventually get caught by a not lazy LLM (fingercoss). Lower the chance of "making story".

How to solve it except prompt engineering? My suggestion is a customized well-tuned LLM just align with our work, and better be RAG one.

- LLM unstable output for complex JSON structure, a more strict and specific *expected_output* example can improve the stability but not guarantee it performance in a large amount of test.

### LLM Choice & Compatiable
A RAG based openai-compatibale-model is highly recommanded in **diagnosis crew**, **possibilities elimination crew** and **treatment plan crew** 
which required highly accurate and professional knowledge to match the tasks' requirement. (Maybe time consuming..)

While, the inquiry crew and multi-agent reprot crew required less accurate and computing power, a fine-tuned LLM which good at summary will be good enough.

## Review and Commen from Author
Generallt speaking, this demo of **Multi-agent Collaborative Rehabilitation Outpatient Assistance System**  is a bold try on whether AI could finally replace doctor after powered by multiple agents and each agent expertised a single work, and imitate a real human doctor's diagnosis logic.

So far, I still think it can be used as a assistance or strong reference for the Rehabilitation Outpatient Doctor, we still have a long way to go to make it thoroughly replace human doctor. But in terms of knowledge amount, information collection and reasoning ability and response speed, there is no doubt we did make a positive difference!

## Further Development
Currently, all the user input gathering and Agent responses are happened in terminal standard I/O, I use lots of *print* to indicate where we are as the flow working.

Those *print* could be very helpful when we actually connect with front-end with this service. We barely need to pass and receive in/output variable in the exactly same place where we *print*.

And there are some agent response print seems not nessecary, but they are actually for the purpose of monitoring the structure of Agents' output, because most likely we will parse these output and extract useful infomation for further usage. 

## Author

**David Qu**  
Undergraduate Researcher | AI Algorithm Engineer

University of Toronto Scarborough - Department of Computer Science \
Yangtze River Delta Guozhi Intelligent Medical Technology Co., Ltd \
Jiangsu Industrial Technology Research Institute \
📧 davidsz.qu@mail.utoronto.ca
