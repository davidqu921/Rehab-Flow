from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class DiagnosisCrew():
    """Rehabilitation DiagnosisCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"  

    @agent
    def diagnosis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['diagnosis_agent'], # type: ignore[index]
            verbose=True
        )
    
    @agent
    def auxiliary_examination_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['auxiliary_examination_agent'], # type: ignore[index]
            verbose=True
        )
    
    @agent
    def diagnosis_quality_control_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['diagnosis_quality_control_agent'], # type: ignore[index]
            verbose=True
        )
    
    @task
    def diagnosis_task(self) -> Task:
        return Task(
            config=self.tasks_config['diagnosis_task'] # type: ignore[index]
        )
    
    @task
    def auxiliary_examination_task(self) -> Task:
        return Task(
            config=self.tasks_config['auxiliary_examination_task'],
            context=[self.diagnosis_task()]
        )
    
    @task
    def diagnosis_quality_control_task(self) -> Task:
        return Task(
            config=self.tasks_config['diagnosis_quality_control_task'], # type: ignore[index]
            context=[self.auxiliary_examination_task()]
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the diagnosis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )