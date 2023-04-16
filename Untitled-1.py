

def check_reach():


def change_stage():
    if finish
        reset()


# random initail block pose, fixed initial hand pose
def reset():

    

def update_state():
    step = torch.max(torch.min((goal - current), mano_dof_upper_limits), mano_dof_lower_limits)

    set_actor_dof_state_tensor_indexed()

    reach_tensor = check_reach()

    change_stage()
