import pandas as pd
import re
import matplotlib.pyplot as plt

import ast






if __name__ == "__main__":
  
    with open("./eval/run_name_mapping.txt", "r", encoding="utf-8") as file:
        content = file.read()

    my_dict = ast.literal_eval(content)

    manual_run_names = ["task_A_1--Julian.json", "task_A_1--Nico.json", "task_A_1--Nils.json", "task_A_1--Ide.json", "task_A_2--Nils.json", "task_A_2--Julian.json", "task_A_2--Nico.json", "task_A_2--Ide.json", "run-webis-expert-manual2-A2.json","run-webis-expert-manual1-A2.json", "task_B--Nils.json", "task_B--Nico.json", "task_B--Julian.json", "task_B--Ide.json"]
    rule_based_run_names = ["run_file_advanced_question_generator_new.json"]
    LLM_persona_run_names = ["task_A_1_automated_gemini-2.5-flash-preview-05-20--UwUGirl373.json", "task_A_1_automated_gemini-2.5-flash-preview-05-20--UwUGirl374.json","task_A_1_automated_gpt-4.1-nano--Nico3.json","task_A_1_automated_gpt-4.1-nano--Nico3.json","task_A_1_automated_gpt-4.1-nano--Nico4.json", "task_A_1_automated_gpt-4.1-nano--Nico5.json", "task_A_1_automated_gpt-4.1-nano--Nico6.json","task_A_2_automated_gemini-2.0-flash--UwUGirl373.json", "task_A_2_automated_gemini-2.0-flash--UwUGirl374.json"
                             "task_A_2_automated_gpt-4.1-nano--Nico3.json", "task_A_2_automated_gpt-4.1-nano--Nico4.json", "task_A_2_automated_gpt-4.1-nano--Nico5.json", "task_A_2_automated_gpt-4.1-nano--Nico6.json",
                             "task_B_automated_gemini-2.0-flash--UwUGirl373.json", "task_B_automated_gemini-2.0-flash--UwUGirl374.json", "task_B_automated_gpt-4.1-nano--Nico3.json", "task_B_automated_gpt-4.1-nano--Nico4.json", "task_B_automated_gpt-4.1-nano--Nico5.json", "task_B_automated_gpt-4.1-nano--Nico6.json"]
    LLM_finetuned_run_names = ["run-webis-gpt2-A1.json", "run-webis-qpt2-A1.json", "run-webis-qpt2-medium-A1.json","run_file_finetuned_block_LLM_new.json", "run_file_finetuned_filter_LLM_new.json", "run-webis-gpt2-A2.json", "run-webis-qpt2-A2.json", "run-webis-qpt2-medium-A2.json"]
    task_choice = input("Choose a task file (A1, A2, B): ")
    file_options = {
        "A1": "eval/results/evaluation_results_A1.csv",
        "A2": "eval/results/evaluation_results_A2.csv",
        "B": "eval/results/evaluation_results_B.csv"
    }

    chosen_file = file_options.get(task_choice, "..eval/results/evaluation_results_A1.csv")
    df = pd.read_csv(chosen_file)


    manual_df = df[df["JSON File"].isin(manual_run_names)]
    rule_df = df[df["JSON File"].isin(rule_based_run_names)]
    LLM_persona_df = df[df["JSON File"].isin(LLM_persona_run_names)]
    LLM_finetuned_df = df[df["JSON File"].isin(LLM_finetuned_run_names)]
    other_df = df[~df["JSON File"].isin(manual_run_names + rule_based_run_names + LLM_persona_run_names + LLM_finetuned_run_names)]

    manual_df["Run Type"] = "Manual"
    rule_df["Run Type"] = "Rule-based"
    LLM_persona_df["Run Type"] = "LLM Persona"
    LLM_finetuned_df["Run Type"] = "LLM Finetuned"
    other_df["Run Type"] = "Other LLM"

    combined_df = pd.concat([manual_df, rule_df, LLM_persona_df, LLM_finetuned_df, other_df], ignore_index=True)
    combined_df["Run Type"] = pd.Categorical(combined_df["Run Type"], categories=["Manual", "Rule-based", "LLM Persona", "LLM Finetuned", "Other LLM"], ordered=True)
    combined_df = combined_df.sort_values(["Run Type", "JSON File"])
    combined_df["Run Type"] = pd.Categorical(combined_df["Run Type"], categories=["Manual", "Other LLM","LLM Persona", "LLM Finetuned", "Rule-based"], ordered=True)

    combined_df = combined_df.sort_values(["Run Type", "JSON File"])
    
    combined_df["JSON File"] = combined_df["JSON File"].apply(lambda x: my_dict.get(x, x))

    fig, ax = plt.subplots(figsize=(14,6))

    colors = {
        "Manual": "#1f77b4",      
        "Other LLM": "#ff7f0e",      
        "Rule-based": "#2ca02c",   
        "LLM Persona": "#d62728", 
        "LLM Finetuned": "#9467bd" 
        }
    
    bar_colors = combined_df["Run Type"].map(colors)

    ax.bar(combined_df["JSON File"], combined_df["Average Rank-Diversity Score"], color=bar_colors)

    ax.set_title("Average Rank-Diversity Score by Run Type")
    ax.set_xlabel("Run Name")
    ax.set_ylabel("Average Rank-Diversity Score")
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3)

    handles = [plt.Rectangle((0,0),1,1, color=colors[rt]) for rt in colors]
    labels = list(colors.keys())
    ax.legend(handles, labels, title="Run Type")

    plt.tight_layout()

    output_file = f"./eval/plots/plot_RDS_{task_choice}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    
