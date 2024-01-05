import numpy as np

def generate_random_walk(num_steps):
    turnings = [np.array([1,1])]
    count_step_turn = 0
    total_step_turn = 100
    turn = False

    for i in range(num_steps - 1):

        if count_step_turn <= total_step_turn:
            sigma_l = turnings[-1][0]
            sigma_r = turnings[-1][1]

        elif turn:
            count_step_turn = 0
            total_step_turn = np.random.gamma(120, 50)
            turn = False
            sigma_l = 1.2
            sigma_r = 1.2
        
        else:
            count_step_turn = 0

            while True:
                sigma_l = np.random.choice([np.random.normal(1, 0.3),
                                                      np.random.normal(1.25, 0.2),
                                                      np.random.normal(0.15, 0.2)],
                                                      p=[0.5, 0.25, 0.25])
                if sigma_l >= 0.0 and sigma_l <= 1.5 : # change before >= 0.1
                    break

            if sigma_l > 1.2:
                proba_r = np.array([0.1, 0.1, 0.8])
            elif (sigma_l > 0.2 and sigma_l < 0.4):
                proba_r = np.array([0.1, 0.8, 0.1])
            elif sigma_l <= 0.2:
                proba_r = np.array([0.2, 0.8, 0.0])
            else:
                proba_r = np.array([0.6, 0.2, 0.2])

            while True:
                sigma_r = np.random.choice([np.random.normal(1, 0.3),
                                                      np.random.normal(1.25, 0.2),
                                                      np.random.normal(0.15, 0.2)],
                                                      p = proba_r)
                if sigma_r >= 0.05 and sigma_l <= 1.7 :
                    break

            if np.abs(sigma_l - sigma_r) > 0.65:
                total_step_turn = np.random.gamma(140, 65)
                turn = True
            else:
                total_step_turn = np.random.gamma(50, 30)
        
        count_step_turn += 1

        turnings.append(np.array([sigma_l,sigma_r]))
        
    return turnings