#!/usr/bin/env python
"""
Author: David Qu
Role: AI Algorithm Engineer 
Affiliation: Yangtze River Delta Guozhi Intelligent Medical Technology Co., Ltd | University of Toronto
Contact: davidsz.qu@mail.utoronto.ca
Project: Multi-Agent Collaborative Rehabilitation Outpatient Assistance System

Description:
This project is developed using the CrewAI framework and implements a multi-agent
collaborative system to assist rehabilitation outpatient care. Multiple intelligent
agents coordinate to handle patient inquiry, medical record analysis, rehabilitation
recommendation, and clinical decision support, enabling seamless human-AI interaction
in the diagnostic workflow.

Key Features:
- Multi-Agent Collaboration: Separate agents for inquiry, auxiliary check suggestions,
  rehabilitation plan generation, etc.
- Flow-Driven Architecture: CrewAI Flow manages task sequences with dynamic branching
  and conditional logic.
- Safety Mechanisms: Maximum iteration limits in loops prevent infinite processing cycles.
- Extensible & Modular: Easily add new agents or workflow nodes to support future
  multi-department applications.

Use Case:
- Rehabilitation outpatient support
- Patient inquiry and information collection
- Personalized rehabilitation recommendation
- Clinical decision assistance and workflow optimization
"""

import json
import os
import uuid
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from rehab_flow.crews.diagnosis_crew.diagnosis_crew import DiagnosisCrew
from rehab_flow.crews.possibilities_elimination_crew.possibilities_elimination_crew import PossibilitiesEliminationCrew
from rehab_flow.crews.treatment_plan_crew.treatment_plan_crew import TreatmentPlanCrew
from rehab_flow.crews.multiagents_report_crew.multiagents_report_crew import MultiAgentsReportCrew
import re

# Define the structure of our diagnosis process outline
class ProcessOutline(BaseModel):
    complete_sections: List[str] = Field(default_factory=list, description="List of completed sections")
    supplementary_inquiries: Dict[str,str] = Field(default_factory=dict, description="Dictionary of supplementary inquiries made during the process")
    suggested_diagnostic_dialectics: List[Dict[str, str]] = Field(default_factory=list, description="List of suggested diagnostic dialectics used in the process") # 建议的辩证分析
    supplementary_auxiliary_examinations: Dict[str, str] = Field(default_factory=dict, description="Dictionary of auxiliary examinations conducted during the process")

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

# Define our flow state
class PatientState(BaseModel):
    initial_inquiry: InitialInquriy = Field(default_factory=InitialInquriy)
    audience_level: str = ""
    process_outline: ProcessOutline = Field(default_factory=ProcessOutline)
    diagnosis_result: Dict[str, str] = {}
    treatment_plan: Dict[str, str] = {}
    # suspended: bool = False  # 后来我认为这部分诊疗过程中的暂停不应该是硬性的，而是简单的页面切换，应该由前端实现


class RehabFlow(Flow[PatientState]):
    """"Rehabilitation Inquiries, Diagnosis and Treatment Flow"""
    
    @start()
    def get_user_input(self):
        """Get input from the user about the initial inquiry and customized audience level"""
        print("\n=== Create Your Comprehensive Diagonsis from Gathering Infomation ===\n")

        print("1. Main complain:")
        self.state.initial_inquiry.main_complain = "骑车摔倒，小臂疼痛，无法活动"
        # self.state.initial_inquiry.main_complain = input("What is the main complaint of the patient? \n")
        print("2. History of present illness:")
        self.state.initial_inquiry.history_of_present_illness = "一年前小臂骨折"
        # self.state.initial_inquiry.history_of_present_illness = input("What is the history of present illness? \n")
        print("3. Past medical history:")
        self.state.initial_inquiry.past_medical_history = "无其他疾病史"
        # self.state.initial_inquiry.past_medical_history = input("What is the past medical history? \n")
        print("4. Allergy history:")
        self.state.initial_inquiry.allergy_history = "无药物过敏史"
        # self.state.initial_inquiry.allergy_history = input("What is the allergy history? \n")
        print("5. Family history:")
        self.state.initial_inquiry.family_history = "无家族遗传病史"
        # self.state.initial_inquiry.family_history = input("What is the family history? \n")
        print("6. Physical examination:")
        self.state.initial_inquiry.physical_examination = "小臂肿胀，压痛明显，活动受限"
        # self.state.initial_inquiry.physical_examination = input("What are the physical examination findings? \n")
        print("7. Personal history:")
        self.state.initial_inquiry.personal_history = "无吸烟饮酒史，工作压力大，缺乏锻炼"
        # self.state.initial_inquiry.personal_history = input("What is the personal history of the patient? \n")
        print("8. Auxiliary examination:")
        self.state.initial_inquiry.auxiliary_examination = {
            "X-ray": "显示小臂骨折愈合不良"
        }
        # print("请每次输入一组检查项目和结果（格式：项目名称=检查结果），输入 q 退出：")
        # while True:
        #     user_input = input("键值对：\n")
        #     if user_input.lower() == 'q':
        #         break
        #     if '=' in user_input:
        #         key, value = user_input.split('=', 1)
        #         self.state.initial_inquiry.auxiliary_examination[key.strip()] = value.strip()
        #     else:
        #         print("❌ 输入格式错误，请使用 key=value 的格式")

        # print("📘 全部检查项目及结果：", self.state.initial_inquiry.auxiliary_examination)

        # Get audience level with validation
        while True:
            print("What is your target audience? (Patient/Professional/Expert)")
            # audience = input("Who is your target audience?(Patient/Professional/Expert) \n").lower()
            audience = "Professional".lower()
            if audience in ["non-professional", "professional", "top-expert"]:
                self.state.audience_level = audience
                break
            print("Please enter 'Non-Profassional', 'Professional', or 'Top-Expert'")

        print(f"\n === 国智康复AI助手已经获得患者信息，主因： {self.state.initial_inquiry.main_complain} \n 使用专业级别为：{self.state.audience_level} === \n")
        return self.state

    @listen(get_user_input)
    def inquiry_assistance(self, p_state):
        """Help to enhence inquiry quality and integrality and build a structured outline for the rehabilitation diagnosis process"""
        print("- 成功解析初诊信息，下面开始智能辅助问诊。")
        # Initialize the supervisor LLM
        llm = LLM(model="openai/glm-4-flash-250414", 
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
                    api_key="d1af5a7ec0ae4a97a2497c262959db20.VJuHoThaguAxjuIx", 
                    temperature=0.2,
                    max_tokens=2000
                    )  # Specify JSON response format
        iter_times = 0
        while "initial_inquiry" not in self.state.process_outline.complete_sections and iter_times < 2: # 这里的loop exit逻辑是较为大胆的交给模型判断的，若是出现infinite loop的情况，可能要hard code一个最大iteration次数
            messages = [
                {"role": "system", "content": "你是一名经验丰富注重细节的康复科门诊主任医师，现在对一位见习医生的实战问诊过程进行监督和指导，对有疑问，不精准，不完整的问诊项目给出提示和补充建议。请确保每次输出为JSON格式."},
                {"role": "user", "content": f"""
                请根据当前值班医生掌握的患者信息, 病患主诉：{p_state.initial_inquiry.main_complain}，现病史：{p_state.initial_inquiry.history_of_present_illness}，
                既往史：{p_state.initial_inquiry.past_medical_history}，过敏史：{p_state.initial_inquiry.allergy_history}，家族史：{p_state.initial_inquiry.family_history}，体格检查：{p_state.initial_inquiry.physical_examination}，
                个人史：{p_state.initial_inquiry.personal_history}，辅助检查：{p_state.initial_inquiry.auxiliary_examination}，已有补充问诊：{self.state.process_outline.supplementary_inquiries}，
                请对值班医生和患者的初问诊内容提出建议，聚焦于患者信息的重点，在明显的信息不完整处，给值班医生提供建议进行的补充提问，
                以实现完整规范的初步诊断流程，帮助缩小可能的初诊结果范围。请注意，目前收集到的患者信息也许为空，""或者为”无”，请结合实际情况提供问诊补充建议即可，不要执着于初诊内容完整性而在当前初诊阶段强行要求提供辅助检查信息，
                更不要在患者不明确现病史、过敏史、家族史，个人史时强行要求补充。初诊阶段，注意力放在收集病人的患病重点信息即可。
                
                回答应包括：
                    1. 对当前初诊内容的分析和总结, 现阶段无需诊断。
                    2. 对值班医生的补充提问建议，回答必须是一个列表，若是无补充建议则返回空列表。
                    3. 判断问诊是否完成，可以进行到下一个任务：正式的诊断推理。仅回答"yes" 或 "no"。
                当辅助初诊流程完成后，输出一个JSON对象，包含以下字段：
                    {{"inquiry_analysis": "当前初诊内容的分析和总结",
                        "supplementary_inquiries": ["对值班医生的补充提问建议1", "对值班医生的补充提问建议2", ...]
                        "inquiry_complete": "yes或是no"
                    }}
                请注意，"supplementary_inquiries"不是必须的，如果当前初诊内容已经足够完整，"supplementary_inquiries"返回空列表[]，"inquiry_complete"回答"yes"。
            
                请确保输出严格遵守上述JSON格式，且不要包含任何多余信息。
                """
                }
            ]

            # Call the LLM to get the response
            print("智能辅助问诊助手正在思考中...")
            response = llm.call(messages=messages)
            # if isinstance(response, str):
            #     print("___ response 是字符串")
            # else:
            #     print("___ response 不是字符串")
            # print(" --- The LLM response of process_outline: ", clean_response)

            def clean_and_parse_json(response: str) -> dict:
                """
                从 Markdown 格式的 response 中提取 JSON 字符串并解析为 dict。
                不修改原始 response，而是生成新的 cleaned_response。
                """
                # 使用正则提取 JSON 块（匹配 ```json 开头 和 ``` 结尾之间的内容）
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
                
                if match:
                    cleaned_response = match.group(1)
                else:
                    # 如果没找到 ``` 包裹的内容，则直接使用原始 response
                    cleaned_response = response.strip()
                
                try:
                    return cleaned_response
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON 解析失败: {e}\n原始内容:\n{cleaned_response}")
            
            clean_response = clean_and_parse_json(response)
       
            if not clean_response.strip():
                raise ValueError("LLM response is empty. Please check the input and try again.")
            # Ensure the response starts with a '{' to be valid JSON
            if not clean_response.strip().startswith('{'):
                raise ValueError("LLM response does not start with '{'. Please check the input and try again.") 

            try:
                supplemetary_inquiry_dict = json.loads(clean_response)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing error: {e}\nResponse content:\n{clean_response}")
            
            # Update the current inquriy analysis
            self.state.process_outline.supplementary_inquiries["inquiry_analysis"] = supplemetary_inquiry_dict.get("inquiry_analysis", "无问诊总结")

            # Print the analysis of the current inquiry
            print(f"=== 当前初诊内容分析 ===\n{supplemetary_inquiry_dict.get('inquiry_analysis', '无分析内容')}\n")

            if len(supplemetary_inquiry_dict.get('supplementary_inquiries', '无补充问诊')) > 0:
                print("*** 推荐补充问诊内容，输入回答后按回车确认，无补充或不采纳直接回车 ***")
                for i in range(len(supplemetary_inquiry_dict.get("supplementary_inquiries"))):  # depends on how many supplementary inquiries are made
                    # Update the state with the parsed suggestions: required to be a dictionary
                    print( f" - 建议补充问诊内容{i+1}：{supplemetary_inquiry_dict.get('supplementary_inquiries')[i]}")
                    supplementary_anwser = input("请输入患者的回复（直接回车表示无补充内容）：\n")
                    
                    if supplementary_anwser != "":
                        # Update the state with the supplementary inquiry
                        self.state.process_outline.supplementary_inquiries[supplemetary_inquiry_dict.get("supplementary_inquiries")[i]] = supplementary_anwser
                        print(f" - 已记录补充问诊：{self.state.process_outline.supplementary_inquiries[supplemetary_inquiry_dict.get('supplementary_inquiries')[i]]} \n 回复: {supplementary_anwser}")
                    else:
                        print(" - 无补充问诊内容被记录")

            # Check if the inquiry is complete
            if supplemetary_inquiry_dict.get("inquiry_complete", "").lower() == "yes":
                self.state.process_outline.complete_sections.append("initial_inquiry")
            iter_times+=1
        print("> 初步问诊已完成，可以进行正式诊断。")
        return self.state.process_outline

    @listen(inquiry_assistance)
    def diagnosis_assistance(self, p_outline):
        """Help to improve the diagonosis accuacy and provide a structured record for the diagnosis process"""
        print("> 成功获得初诊信息，下面开始智能辅助诊断。")
        print("--- 输入回车开始智能辅助诊断，如要跳过智能辅助诊断流程，请输入q ---")
        if input("确认跳过智能辅助诊断流程？: ").lower() == 'q':
            # Update the state with the final diagnosis result
            final_diagnosis_conclusion = input("请手动输入诊断结论：\n")
            final_diagnosis_basis = input("请手动输入诊断依据：\n")
            self.state.diagnosis_result = {
                "diagnosis_conclusion": final_diagnosis_conclusion,
                "diagnosis_basis": final_diagnosis_basis
            }
            # Store the diagnosis result in the process outline
            self.state.process_outline.complete_sections.append("diagnosis")        
            print("> 智能诊断助手已跳过，直接进入下一步治疗方案制定。")
            return self.state.diagnosis_result
        
        else:
            print("> 智能诊断助手正在分析患者信息...")
            audience_level = self.state.audience_level
            main_complain = self.state.initial_inquiry.main_complain
            history_of_present_illness = self.state.initial_inquiry.history_of_present_illness
            past_medical_history = self.state.initial_inquiry.past_medical_history
            allergy_history = self.state.initial_inquiry.allergy_history
            family_history = self.state.initial_inquiry.family_history
            physical_examination = self.state.initial_inquiry.physical_examination
            personal_history = self.state.initial_inquiry.personal_history
            auxiliary_examination = self.state.initial_inquiry.auxiliary_examination

            supplementary_inquiries = p_outline.supplementary_inquiries
            
            # Call the DiagnosisCrew to get the diagnosis result
            result = DiagnosisCrew().crew().kickoff(inputs={  # 这个kickoff应当是在Crewbase里面的父类方法
                "audience_level": audience_level,
                "main_complain": main_complain,
                "history_of_present_illness": history_of_present_illness,
                "past_medical_history": past_medical_history,
                "allergy_history": allergy_history,
                "family_history": family_history,
                "physical_examination": physical_examination,
                "personal_history": personal_history,
                "auxiliary_examination": auxiliary_examination,
                "supplementary_inquiries": supplementary_inquiries
            })

            print("=== 智能诊断助手分析结果 ===")
            
            print(result.raw)  # Crew生成的内容

            # 解析LLM输出是必要的步骤！
            parsed_result = json.loads(result.raw)

            print("--- 当前诊断结论 ---")
            diagnosis_conclusion = parsed_result.get("diagnosis_conclusion", "无诊断结论")
            print(diagnosis_conclusion)
            print("--- 诊断依据 ---")
            diagnosis_basis = parsed_result.get("diagnosis_basis", "无诊断依据")
            print(diagnosis_basis)
            print("--- 需要鉴别(存疑)的疾病名称 ---")
            list_of_other_diagnosis_possibility = parsed_result.get("other_diagnosis_possibility", [])
            for i in range(len(list_of_other_diagnosis_possibility)):
                print(f" 第 {i+1} 种可能 -\n", list_of_other_diagnosis_possibility[i])
            
            str_of_other_diagnosis_possibility = "; ".join(
                    ", ".join(f"{k}:{v}" for k, v in entity.items())
                    for entity in list_of_other_diagnosis_possibility
                )
            print("--- 智能诊断质控反馈 ---")
            quality_control_feedback = parsed_result.get("quality_control_feedback", "无质控反馈")
            print(quality_control_feedback)

            print("--- 建议问题缩小诊断范围 ---")
            print(parsed_result.get("suggested_question", "无建议问题"))
            answer_suggested_question = input("请回答建议问题（直接回车表示无补充内容）：\n")
            if answer_suggested_question != "":
                new_suggested_question_and_answer = { parsed_result.get("suggested_question", "无建议问题"): answer_suggested_question }
                self.state.process_outline.suggested_diagnostic_dialectics.append(new_suggested_question_and_answer)
                print(f" - 已记录补充诊断对话：{parsed_result.get('suggested_question', '无建议问题')} \n 回复: {answer_suggested_question}")
            else:
                print(" - 无补充问诊内容被记录")

            # Check if there are suggested auxiliary checks
            if len(parsed_result.get("suggested_auxiliary_examinations", [])) != 0:
                print("--- 建议的辅助检查项目 ---") # 想添加可以添加， 无需添加时可跳过
                for i in range(len(parsed_result.get("suggested_auxiliary_examinations", []))):
                    print(f" 第 {i+1} 项 - ", parsed_result.get("suggested_auxiliary_examinations", [])[i])
                    print("*** 推荐补充辅助检查，输入检查结果后按回车确认，无补充或不采纳直接回车 ***")
                    suggested_auxiliary_check_answer = input("请输入检查结果（直接回车表示无补充内容）：\n")
                    if suggested_auxiliary_check_answer != "":
                        # ///////////////////////改这里//////////////////////////parsed_result.get("suggested_auxiliary_examinations")[i]是一个字典，不能当作key
                        self.state.process_outline.supplementary_auxiliary_examinations[parsed_result.get("suggested_auxiliary_examinations")[i].get("examination_name","无检查项目")] = suggested_auxiliary_check_answer
                        print(f" - 已记录补充辅助检查：{parsed_result.get('suggested_auxiliary_examinations', [])[i]} \n 检查结果: {suggested_auxiliary_check_answer}")
                    else:
                        print(" - 无补充辅助检查内容被记录")


            if len(self.state.process_outline.suggested_diagnostic_dialectics) > 0:
                suggested_diagnostic_dialectics = self.state.process_outline.suggested_diagnostic_dialectics[-1] # 如有报错，可尝试将list整理成str输入模型
            else:
                suggested_diagnostic_dialectics = "暂无建议的诊断对话"
            
            # 现在有：存疑的疾病可能，一套suggested_diagnostic_dialectics问题和回答，建议的辅助检查项目和结果
            # 我们接下来在此处使用possibilities_eliminator_crew来分析得到的最新信息来排除存疑的可能性

            if len(list_of_other_diagnosis_possibility) > 0:
                ready_to_treatment_plan = False
                while  ready_to_treatment_plan != True:
                    print("=== 智能诊断助手正在进一步排除存疑的疾病可能性 ===")
                    # Call the PossibilitiesEliminationCrew to get the elimination result
                    # 通过循环来实现多轮排除直至确诊
                    elimination_result = PossibilitiesEliminationCrew().crew().kickoff(inputs={
                        "diagnosis_conclusion": diagnosis_conclusion,
                        "diagnosis_basis": diagnosis_basis,
                        "other_diagnosis_possibility": str_of_other_diagnosis_possibility,
                        "suggested_question_and_answer": suggested_diagnostic_dialectics, # 这里是一个dict[str,str]
                        "suggested_auxiliary_check_and_result": self.state.process_outline.supplementary_auxiliary_examinations,
                        "audience_level": audience_level
                    })

                    print(elimination_result.raw)
                    # 解析LLM的输出是必要的！
                    parsed_elimination_result = json.loads(elimination_result.raw)

                    print("- 是否需要进一步辅助诊断：", parsed_elimination_result.get("further_inquiries_needed", "unknown"))
                    if parsed_elimination_result.get("further_inquiries_needed", "unknown").lower() == "no":
                        print(" > 无需进一步辅助诊断，诊断已完成。")
                        print("--- 最终诊断结论 ---")
                        diagnosis_conclusion = parsed_elimination_result.get("diagnosis_conclusion", "无更新诊断结论")
                        print(diagnosis_conclusion)
                        print("--- 最终诊断依据 ---")
                        diagnosis_basis = parsed_elimination_result.get("diagnosis_basis", "无更新诊断依据")
                        print(diagnosis_basis)
                        ready_to_treatment_plan = True
                    else:
                        print(" > 需要进一步辅助诊断，继续排除存疑的疾病可能性。")
                        print("--- 更新诊断结论 ---")
                        diagnosis_conclusion = parsed_elimination_result.get("diagnosis_conclusion", "无更新诊断结论")
                        print(diagnosis_conclusion)
                        print("--- 更新诊断依据 ---")
                        diagnosis_basis = parsed_elimination_result.get("diagnosis_basis", "无更新诊断依据")
                        print(diagnosis_basis)
                        print("--- 需要进一步排除的其他可能的诊断 ---")
                        list_of_other_diagnosis_possibility = parsed_elimination_result.get("other_diagnosis_possibility", [])
                        
                        # 如果没有其他可能的诊断，说明诊断已经完成
                        if len(list_of_other_diagnosis_possibility) == 0:
                            print(" > 无其他可能的诊断，诊断已完成。")
                            ready_to_treatment_plan = True
                        else:
                            for i in range(len(list_of_other_diagnosis_possibility)):
                                print(f" 第 {i} 种可能 -\n", list_of_other_diagnosis_possibility[i])
                            str_of_other_diagnosis_possibility = "; ".join(
                                ", ".join(f"{k}:{v}" for k, v in entity.items())
                                for entity in list_of_other_diagnosis_possibility
                            )
                            
                            print("--- 需要进一步的辅助检查 ---")
                            if len(parsed_elimination_result.get("suggested_auxiliary_check", [])) == 0:
                                print(" > 无建议的辅助检查。")
                                ready_to_treatment_plan = True
                            else:
                                for i in range(len(parsed_elimination_result.get("suggested_auxiliary_check", []))):
                                    print("*** 推荐补充辅助检查，输入检查结果后按回车确认，无补充或不采纳直接回车 ***")
                                    print(f" 第 {i} 项 -\n", parsed_elimination_result.get("suggested_auxiliary_check", [])[i])
                                    suggested_auxiliary_check_answer = input("请输入检查结果（直接回车表示无补充内容）：\n")
                                    if suggested_auxiliary_check_answer != "":
                                        self.state.process_outline.supplementary_auxiliary_examinations[parsed_elimination_result.get("suggested_auxiliary_check", [])[i]] = suggested_auxiliary_check_answer
                                        print(f" - 已记录补充辅助检查：{parsed_elimination_result.get('suggested_auxiliary_check', [])[i]} \n 检查结果: {suggested_auxiliary_check_answer}")
                                    else:
                                        print(" - 无补充辅助检查内容被记录")

                            print("--- 需要进一步追问的症状 ---")
                            suggested_diagnostic_question = parsed_elimination_result.get("suggested_question", "无建议问题")
                            answer_suggested_question = input("请回答建议问题（直接回车表示无补充内容，或者无建议问题）：\n")
                            if answer_suggested_question != "":
                                suggested_diagnostic_dialectics = { suggested_diagnostic_question: answer_suggested_question }
                                self.state.process_outline.suggested_diagnostic_dialectics.append(suggested_diagnostic_dialectics)
                                print(f" - 已记录补充诊断对话：{parsed_elimination_result.get('suggested_question', '无建议问题')} \n 回复: {answer_suggested_question}")
                            else:
                                print(" - 无补充问诊内容被记录")
                                suggested_diagnostic_dialectics = {}            
            else:
                print(" > 无存疑的疾病可能，诊断已完成。")
            
            final_diagnosis_conclusion = diagnosis_conclusion
            final_diagnosis_basis = diagnosis_basis
            
            # Update the state with the final diagnosis result
            self.state.diagnosis_result = {
                "diagnosis_conclusion": final_diagnosis_conclusion,
                "diagnosis_basis": final_diagnosis_basis
            }
            # Store the diagnosis result in the process outline
            self.state.process_outline.complete_sections.append("diagnosis")
            print("> 诊断已完成，可以进行制定治疗方案。")
        
        return self.state.diagnosis_result

    ###########################################################################

    @listen(diagnosis_assistance)
    def treatment_plan_creation(self, p_diagnosis_result):
        """Create a treatment plan based on the diagnosis result"""
        print("> 成功获得诊断结果，下面开始制定治疗方案。")
        print("--- 跳过智能辅助治疗方案制定流程，请输入 q，不跳过，请输入回车 ---")
        treatment_plan_filename = str(uuid.uuid4()) # Generate a unique filename for the treatment plan
        if input("确认跳过智能辅助治疗方案制定流程？: ").lower() == 'q':
            # Update the state with the final treatment plan
            treatment_plan = input("请手动输入治疗方案：\n")
            self.state.treatment_plan["manual_treatment_plan"] = treatment_plan
            
            with open(f"treatment_plans/treatment_plan_{treatment_plan_filename}.md", "w") as f:
                f.write(treatment_plan)
            print(f"> 手写治疗方案已保存在文档 treatment_plans/treatment_plan_{treatment_plan_filename}.md中。")
            print("> 智能治疗方案助手已跳过。")

        else:
            print("> 智能治疗方案助手正在分析诊断结果...")
            audience_level = self.state.audience_level
            diagnosis_conclusion = p_diagnosis_result.get("diagnosis_conclusion", "无诊断结论")
            diagnosis_basis = p_diagnosis_result.get("diagnosis_basis", "无诊断依据")
            supplementary_auxiliary_examinations = self.state.process_outline.supplementary_auxiliary_examinations
            suggested_diagnostic_dialectics = self.state.process_outline.supplementary_auxiliary_examinations
            
            history_of_present_illness = self.state.initial_inquiry.history_of_present_illness
            past_medical_history = self.state.initial_inquiry.past_medical_history
            allergy_history = self.state.initial_inquiry.allergy_history
            family_history = self.state.initial_inquiry.family_history
            personal_history = self.state.initial_inquiry.personal_history

            # Call the TreatmentPlanCrew to get the treatment plan
            plan_result = TreatmentPlanCrew().crew().kickoff(inputs={  # 这个kickoff应当是在Crewbase里面的父类方法
                "audience_level": audience_level,
                "diagnosis_conclusion": diagnosis_conclusion,
                "diagnosis_basis": diagnosis_basis,
                "supplementary_auxiliary_examinations": supplementary_auxiliary_examinations,
                "suggested_diagnostic_dialectics": suggested_diagnostic_dialectics,
                
                "history_of_present_illness": history_of_present_illness,
                "past_medical_history": past_medical_history,
                "allergy_history":  allergy_history,
                "family_history": family_history,
                "personal_history": personal_history
            })

            print("=== 智能治疗方案助手分析结果 ===")
            print(plan_result.raw)
            
            # 目前的治疗计划输出的具体结构是由大模型自己决定的，人为确定大体上是：markdown/json/str 的输出类型
            # 注意： 目前跟他说输出JSON，会有不稳定的非法JSON输出的可能，导致后面解析失败,除非JSON内部的具体格式很明确，且不太复杂 （大JSON套小JSON十有八九要出问题）
            try:
                dict_plan_result = json.loads(plan_result.raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing error: {e}")
            
            self.state.treatment_plan.update(dict_plan_result)
            print("> 智能治疗方案助存入State")

            with open(f"treatment_plans/treatment_plan_{treatment_plan_filename}.md", "w") as f:
                f.write(plan_result.raw)
            print(f"> 智能治疗方案已保存在文档 treatment_plans/treatment_plan_{treatment_plan_filename}.md中。")
        
        print("> 多智能体辅助治疗计划以完成，进入下一步：诊断报告生成！")
        # Store the treatment plan in the process outline
        self.state.process_outline.complete_sections.append("treatment plan")
        return self.state.process_outline
    
    
    @listen(treatment_plan_creation)
    def multiagents_report(self, p_outline):
        """Write a final report based on the full process record and multi-agent cooperation dialogue"""
        
        def prepare_for_llm(list_of_dict):
            """
            将 List[Dict[str, str]] 转换为 LLM 可稳定解析的 JSON 字符串
            """
            # 确保格式正确，中文不转义
            return json.dumps(
                list_of_dict,
                ensure_ascii=False,  # 保留中文
                indent=2             # 美化缩进
            )
        
        # 尽量将输入模型的变量转化成str
        audience_level = self.state.audience_level
        
        # 初诊信息
        history_of_present_illness = self.state.initial_inquiry.history_of_present_illness
        past_medical_history = self.state.initial_inquiry.past_medical_history
        allergy_history = self.state.initial_inquiry.allergy_history
        family_history = self.state.initial_inquiry.family_history
        personal_history = self.state.initial_inquiry.personal_history
        physical_examination = self.state.initial_inquiry.physical_examination
        personal_history = self.state.initial_inquiry.personal_history
        auxiliary_examination = self.state.initial_inquiry.auxiliary_examination # 这是一个dict，若是模型理解不好可以先解析成JSON再传入

        # 后续补充问诊和诊断时的辩证
        supplementary_inquiries = p_outline.supplementary_inquiries  # 这是一个dict
        suggested_diagnostic_dialectics = prepare_for_llm(p_outline.suggested_diagnostic_dialectics)
        supplementary_auxiliary_examinations = p_outline.supplementary_auxiliary_examinations # 这是一个dict
        complete_sections = prepare_for_llm(p_outline.complete_sections)

        # 诊断结果
        diagnosis_result = self.state.diagnosis_result # 这是一个dict
        
        # 治疗计划
        treatment_plan = self.state.treatment_plan # 这是一个dict

        final_report = MultiAgentsReportCrew().crew().kickoff(inputs={  # 这个kickoff应当是在Crewbase里面的父类方法
                "audience_level": audience_level,

                "history_of_present_illness": history_of_present_illness,
                "past_medical_history": past_medical_history,
                "allergy_history":  allergy_history,
                "family_history": family_history,
                "personal_history": personal_history,
                "physical_examination": physical_examination,
                "auxiliary_examination": auxiliary_examination,

                "supplementary_inquiries": supplementary_inquiries,
                "suggested_diagnostic_dialectics": suggested_diagnostic_dialectics,
                "supplementary_auxiliary_examinations": supplementary_auxiliary_examinations,
                "complete_sections": complete_sections,

                "diagnosis_result": diagnosis_result,
                "treatment_plan": treatment_plan
            })
        
        print("=== 多智能体协作辅助诊断总结 ===")
        print(final_report.raw)

        report_filename = str(uuid.uuid4())

        with open(f"summary_reports/report_{report_filename}.md", "w") as f:
                f.write(final_report.raw)
        print(f"> 多智能体协作辅助诊断总结已保存在文档 summary_reports/report_{report_filename}.md中。")
        return "Summary of multi-agents cooperating to diagonse is done"

def kickoff():
    """Run the rehab flow"""
    RehabFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive treatment plan and multi-agent diagnosis report is ready in the treatment_plans directory and multiagents_report directory.")
    print("~ Mission Complished - Go Conquer the World, David!")

def plot():
    """Generate a visualization of the flow"""
    flow = RehabFlow()
    flow.plot("rehab_flow_1")
    print("Flow visualization saved to rehab_flow.html")

if __name__ == "__main__":
    kickoff()