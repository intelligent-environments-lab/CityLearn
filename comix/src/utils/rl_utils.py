import torch as th

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    bs = rewards.size(0)
    max_t = rewards.size(1)
    targets = rewards.new(target_qs.size()).zero_()
    running_target = rewards.new(bs, n_agents).zero_()
    terminated = terminated.float()
    for t in reversed(range(max_t)):
        if t == max_t - 1:
            running_target = mask[:, t] * (rewards[:, t] + gamma * (1 - terminated[:, t]) * target_qs[:, t])
        else:
            running_target = mask[:, t] * (
                terminated[:, t] * rewards[:, t]
                + (1 - terminated[:, t]) * (rewards[:, t] + gamma * (
                               td_lambda * running_target
                               + (1 - td_lambda) * target_qs[:, t])
                                           ))
        targets[:, t, :] = running_target
    return targets

def preprocess_scheme(scheme, preprocess):

    if preprocess is not None:
        for k in preprocess:
            assert k in scheme
            new_k = preprocess[k][0]
            transforms = preprocess[k][1]

            vshape = scheme[k]["vshape"]
            dtype = scheme[k]["dtype"]
            for transform in transforms:
                vshape, dtype = transform.infer_output_info(vshape, dtype)

            scheme[new_k] = {
                "vshape": vshape,
                "dtype": dtype
            }
            if "group" in scheme[k]:
                scheme[new_k]["group"] = scheme[k]["group"]
            if "episode_const" in scheme[k]:
                scheme[new_k]["episode_const"] = scheme[k]["episode_const"]

    return scheme