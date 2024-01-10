from libs.payload_capability import solve_maximum_payload

import os
import numpy as np
import matplotlib.pyplot as plt

# イプシロンロケットの値（参考文献：https://global.jaxa.jp/projects/rockets/epsilon/pdf/EpsilonUsersManual_e.pdf）
Isp_list = [284.0, 300.0, 301.0] # [s]
first_stage_weight = 75.0 # [ton]
first_stage_prop = 66.3
second_stage_weight = 17.0
second_stage_prop = 15.0
third_stage_weight = 3.3
third_stage_prop = 2.5

M_i_list = [
    first_stage_weight + second_stage_weight + third_stage_weight,
    second_stage_weight + third_stage_weight,
    third_stage_weight   
]
M_f_list = [
    first_stage_weight - first_stage_prop + second_stage_weight + third_stage_weight,
    second_stage_weight - second_stage_prop + third_stage_weight,
    third_stage_weight - third_stage_prop  
]

C3_list = np.linspace(5, 140, 1000) # [km^2/s^2]
M_L_list = []
for C3 in C3_list:
    maximum_payload = solve_maximum_payload(Isp_list=Isp_list, M_i_list=M_i_list, M_f_list=M_f_list, C3=C3)
    M_L_list.append(maximum_payload)

plt.plot(C3_list, M_L_list)
plt.ylabel("$M_L$ [ton]")
plt.xlabel("$C_3 [km^2/s^2]$")
os.makedirs("outputs/", exist_ok=True)
plt.savefig("outputs/payload_capability.png")