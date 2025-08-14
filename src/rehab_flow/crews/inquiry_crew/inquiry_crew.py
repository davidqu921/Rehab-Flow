from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class InquiryCrew:
    """Rehabilitation Inquriy Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def inquiry_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['inquiry_agent'],
            verbose=True
        )
    
    @agent
    def inquiry_quality_control_agent(self)-> Agent:
        return Agent(
            config=self.agents_config['inquiry_quality_control_agent'],
            verbose=True
        )
    
    @task
    def inquiry_task(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_task']
        )
    
    @task
    def inquiry_quality_control_task(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_quality_control_task'],
            context=[self.inquiry_task()]
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the Rehablitation Inquiry Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )