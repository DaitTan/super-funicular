import pickle
from driver import Objectives, Population, GARoutine, NonDominatedSorting
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pathlib
from plotly.subplots import make_subplots


def obj_1(pop):
    x = pop[:, 0]

    return -1 * (x * (np.sin(10 * np.pi * x)))

def obj_2(pop):
    x = pop[:, 0]
    return -1 * (2.5 * x * (np.cos(3 * np.pi * x)))


objective_1 = {}
objective_1["type"] = "Minimize"
objective_1["function"] = obj_1

objective_2 = {}
objective_2["type"] = "Minimize"
objective_2["function"] = obj_2

true_obj_x = np.linspace(0,1, 1000).reshape(-1,1)
true_obj_1_y = -1*obj_1(true_obj_x)
true_obj_2_y = -1*obj_2(true_obj_x)


obj_1_true_df = pd.DataFrame(dict(
                    x = true_obj_x[:,0],
                    y = true_obj_1_y,
                ))
obj_2_true_df = pd.DataFrame(dict(
                    x = true_obj_x[:,0],
                    y = true_obj_2_y,
                ))

objectives_list = [objective_1, objective_2]
objectives = Objectives(objectives_list)

# def obj_1(pop):
#     x = pop[:, 0]

#     return x**2

# def obj_2(pop):
#     x = pop[:, 0]
#     return (x-2)**2


# objective_1 = {}
# objective_1["type"] = "Minimize"
# objective_1["function"] = obj_1

# objective_2 = {}
# objective_2["type"] = "Minimize"
# objective_2["function"] = obj_2

# true_obj_x = np.linspace(0,1, 1000).reshape(-1,1)
# true_obj_1_y = obj_1(true_obj_x)
# true_obj_2_y = obj_2(true_obj_x)


# obj_1_true_df = pd.DataFrame(dict(
#                     x = true_obj_x[:,0],
#                     y = true_obj_1_y,
#                 ))
# obj_2_true_df = pd.DataFrame(dict(
#                     x = true_obj_x[:,0],
#                     y = true_obj_2_y,
#                 ))

# objectives_list = [objective_1, objective_2]
# objectives = Objectives(objectives_list)

obj_name = "test_ex"
start_seed = 123458

population_size = 100
num_variables = 1
bounds = [[0,1]]*1

crossover_prob = 0.9
mutation_prob = 0.15

p_curve_param = 20
p_curve_param_mutation = 20

num_generations = 100
num_runs = 1


base_path = pathlib.Path()
result_directory = base_path.joinpath(obj_name+"_images")
result_directory.mkdir(exist_ok=True)

for runs in range(num_runs):
    seed = start_seed + runs
    run_folder = result_directory.joinpath(obj_name + "_images_Run_"+str(runs) + "_seed_"+str(seed))
    run_folder.mkdir(exist_ok=True)
    for gen in range(num_generations):
        
        file_name = "{}/{}_Run_{}_seed_{}/{}_Gen_{}_Run_{}.pkl".format(obj_name, obj_name, runs, seed, obj_name, gen, runs)
        # print(file_name)
        # SCH/SCH_Run_0_seed_123458/SCH_Gen_0_Run_0.pkl
        f = open(file_name, "rb")
        arr = pickle.load(f)
        print(arr.shape)
        pop = Population(population_size, num_variables, bounds, objectives, seed, arr, False)
        f.close()
        
        fig = make_subplots(
            rows=2, cols=2, 
            specs=[[{"rowspan": 2},{} ],
            [None, {}]],
            print_grid=True)
        all_frontiers = NonDominatedSorting(pop)
        
        for rank, frontiers in enumerate(all_frontiers.all_frontiers):
            evaluations = []
            sr_num = []
            for iterate in frontiers.points_in_frontier:
                point,_ = pop.fetch_by_serial_number(iterate)
                evaluations.append(point.corres_eval)
                sr_num.append(point.serial_number)
            evaluations = np.array(evaluations)
            df = pd.DataFrame(dict(
                    x = -1*evaluations[:,0],
                    y = -1*evaluations[:,1],
                ))
            
            point_caption = (["Point {}".format(i) for i in sr_num])
            fig.add_trace(go.Scatter(
                x = df.sort_values(by="x")["x"],
                y = df.sort_values(by="x")["y"],
                mode = "markers+lines",
                text = point_caption,
                name = "Frontier {}".format(rank + 1),
                showlegend=False,
                marker=dict(
                size=2,
                line=dict(
                    width=1
                    )
                ),
                
            ),
            row = 2, col = 2
            )

        # fig.update_yaxes(
        #     scaleanchor = "x",
        #     scaleratio = 1,
        # )
        # fig.update_xaxes(
        #     scaleanchor = "x",
        #     scaleratio = 1,
        # )

        all_sol_vec = pop.get_all_sol_vecs()
        all_evaluations = pop.get_all_evals()


        #####################################################################
        fig.add_trace(go.Scatter(
            x = obj_1_true_df.sort_values(by="x")["x"],
            y = obj_1_true_df.sort_values(by="x")["y"],
            mode = "lines",
            text = point_caption,
            name = "Objective 1",
            showlegend=False,
            ),
            row = 1, col = 1
            )
        
        fig.add_trace(go.Scatter(
        x = all_sol_vec[:,0],
        y = -1*all_evaluations[:,0],
        mode = "markers",
        name = "Solutions on Obj. 1",
        showlegend=False,
        ),
        row = 1, col =1
        )

        fig.add_trace(go.Scatter(
            x = obj_2_true_df.sort_values(by="x")["x"],
            y = obj_2_true_df.sort_values(by="x")["y"],
            mode = "lines",
            name = "Objective 2",
            showlegend=False,
            ),
            row = 1, col = 1
            )
        all_evaluations = pop.get_all_evals()
        fig.add_trace(go.Scatter(
        x = all_sol_vec[:,0],
        y = -1*all_evaluations[:,1],
        mode = "markers",
        name = "Solutions on Obj. 2",
        showlegend=False,
        ),
        row = 1, col = 1
        )
        #####################################################################
        df = pd.DataFrame(dict(
            x = true_obj_1_y,
            y = true_obj_2_y,
        ))
        
        fig.add_trace(go.Scatter(
        x = df.sort_values(by="x")["x"],
        y = df.sort_values(by="x")["y"],
        marker=dict(
            size=2),
        mode = "markers",
        name = "True Obj1 vs Obj2",
        showlegend=False,
        ),
        row = 1, col = 2
        )


        df = pd.DataFrame(dict(
            x = -1*all_evaluations[:,0],
            y = -1*all_evaluations[:,1],
        ))

        fig.add_trace(go.Scatter(
        x = df.sort_values(by="x")["x"],
        y = df.sort_values(by="x")["y"],
        mode = "markers",
        name = "Best Solutions Obj1 vs Obj2",
        showlegend=False,
        ),
        row = 1, col = 2
        )
        #####################################################################
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_xaxes(title_text="Objective 1", row=1, col=2)
        fig.update_xaxes(title_text="Objective 1", row=2, col=2)

        fig.update_yaxes(title_text="f(x)", row=1, col=1)
        fig.update_yaxes(title_text="Objective 2", row=1, col=2)
        fig.update_yaxes(title_text="Objective 2", row=2, col=2)
        # fig['layout']['xaxis']['title']='x'
        # fig['layout']['xaxis2']['title']='Objective 1'
        # fig['layout']['xaxis3']['title']='Objective 1'
        # fig['layout']['yaxis']['title']='f(x)'
        # fig['layout']['yaxis2']['title']='Objective 2'
        # fig['layout']['yaxis3']['title']='Objective 2'
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)'
        )
        im_file_name = run_folder.joinpath("{}_Gen_{}_Run_{}.png".format(obj_name, gen, runs))
        
        fig.write_image(str(im_file_name))
        # print(fjifbhibf)

