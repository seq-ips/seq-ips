Source code and experiments of simulation data 

Data generated with 2 simulation seeds and pre-trained exposure models are included, debiased rating prediction experiments can be run as:  
For Naive:  
python3 main_debiasing_simulation.py --dataset simulate_w0 --if_weight 0  
For Pop:  
python3 main_debiasing_simulation.py --dataset simulate_w0 --if_weight 1 --weight_type 0  
For PF:  
python3 main_debiasing_simulation.py --dataset simulate_w0 --if_weight 1 --weight_type 1   
For our model:  
python3 main_debiasing_simulation.py --dataset simulate_w0 --if_weight 1 --weight_type 2   

The whole pipeline of our method can be run as:  
Generate new simulation data:  
python3 generate_simulation_data.py --dataset_name simulate_w2   
Train exposure model:  
python3 train_seq_expo_model_simulation.py --dataset simulate_w2
python3 calculate_seqexpo_propensity_simulation.py --dataset simulate_w2
Debiased rating prediction:  
python3 main_debiasing_simulation.py --dataset simulate_w2 --if_weight 1 --weight_type 2   

For PF model:  
python3 calculate_hpf_propensity_simulation.py --dataset simulate_w2  
python3 main_debiasing_simulation.py --dataset simulate_w2 --if_weight 1 --weight_type 1 
