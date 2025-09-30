
###CREMA-D Res
python train.py --config ./configs/CREMA_D/release/res/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/ens.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/joint_training.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/multiloss.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/pre_finetuned.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/pre_frozen.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/CREMA_D/release/res/AGM.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 3.0
python train.py --config ./configs/CREMA_D/release/res/OGM.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 1.0
python train.py --config ./configs/CREMA_D/release/res/MLB.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 0.5 --tanh_mode_beta 10
python train.py --config ./configs/CREMA_D/release/res/MSLR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001 --kmepoch 20 --ilr_c 0.7 --ilr_g 1.3
python train.py --config ./configs/CREMA_D/release/res/MMCosine.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001 --mmcosine_scaling 10
python train.py --config ./configs/CREMA_D/release/res/DnR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 3.0 --kmepoch 3
python train.py --config ./configs/CREMA_D/release/res/MMPareto.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha 1.5
python train.py --config ./configs/CREMA_D/release/res/ReconBoost.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha 0.5 --recon_weight1 3 --recon_weight2 1 --recon_epochstages 1 --recon_ensemblestages 1
python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre

#Ablations
python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by ind --batch_size 32 --contr_coeff 1 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by colab --batch_size 32 --contr_coeff 1 --pre

python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 0 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 0 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre  --lib 1

python train.py --config ./configs/CREMA_D/release/res/MCR_NoiseInput.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 8 --reg_by greedy --batch_size 8 --contr_coeff 1 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR_ZeroInput.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR_NoiseLatent.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre
python train.py --config ./configs/CREMA_D/release/res/MCR_ZeroLatent.json --default_config ./configs/CREMA_D/default_config_cremad_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre

####CREMA-D Vit
python train.py --config ./configs/CREMA_D/release/vit/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/ens.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/joint_training.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/multiloss.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/pre_finetuned.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/pre_frozen.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/AGM.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --alpha 3.0
python train.py --config ./configs/CREMA_D/release/vit/OGM.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --alpha 1.0
python train.py --config ./configs/CREMA_D/release/vit/MLB.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --alpha 5.0
python train.py --config ./configs/CREMA_D/release/vit/MSLR.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/MMCosine.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6
python train.py --config ./configs/CREMA_D/release/vit/DnR.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --alpha 3.0 --kmepoch 3 --batch_size 8
python train.py --config ./configs/CREMA_D/release/vit/MMPareto.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --alpha 2.0 --batch_size 8
python train.py --config ./configs/CREMA_D/release/vit/ReconBoost.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --alpha 0.5 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 4 --recon_ensemblestages 4 --batch_size 8
python train.py --config ./configs/CREMA_D/release/vit/MCR.json --default_config ./configs/CREMA_D/default_config_cremad_vit.json --fold 0 --lr 0.00005 --wd 5e-6 --l 0.1 --multil 1 --num_samples 32 --reg_by greedy --batch_size 8 --contr_coeff 1 --lib 1

###AVE Res
python train.py --config ./configs/AVE/release/res/unimodal_audio.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0
python train.py --config ./configs/AVE/release/res/unimodal_video.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0
python train.py --config ./configs/AVE/release/res/ens.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0
python train.py --config ./configs/AVE/release/res/joint_training.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/AVE/release/res/multiloss.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/AVE/release/res/pre_frozen.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/AVE/release/res/pre_finetuned.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001
python train.py --config ./configs/AVE/release/res/MSLR.json --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --lr 0.001 --wd 0.0001 --kmepoch 10 --ilr_c 0.7 --ilr_g 1.3
python train.py --config ./configs/AVE/release/res/MMCosine.json --default_config ./configs/AVE/default_config_ave_res.json --fold 1 --lr 0.001 --wd 0.0001 --mmcosine_scaling 1
python train.py --config ./configs/AVE/release/res/OGM.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --alpha 0.6 --lr 0.001 --wd 0.0001
python train.py --config ./configs/AVE/release/res/AGM.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --alpha 3.0 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/AVE/release/res/MLB.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 2.0 --tanh_mode_beta 1
python train.py --config ./configs/AVE/release/res/MMPareto.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 0.5
python train.py --config ./configs/AVE/release/res/ReconBoost.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 0.5 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 1 --recon_ensemblestages 1
python train.py --config ./configs/AVE/release/res/DnR.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001 --alpha 3.0 --kmepoch 3
python train.py --config ./configs/AVE/release/res/MCR.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.01 --multil 0.01 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre

###AVE Conformer
python train.py --config ./configs/AVE/release/vit/unimodal_video.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/AVE/release/vit/unimodal_audio.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/AVE/release/vit/ens.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/AVE/release/vit/pre_finetuned.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.00001 --wd 0.0001 --batch_size 8
python train.py --config ./configs/AVE/release/vit/pre_frozen.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.00001 --wd 0.0001 --batch_size 8
python train.py --config ./configs/AVE/release/vit/joint_training.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8
python train.py --config ./configs/AVE/release/vit/multiloss.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8
python train.py --config ./configs/AVE/release/vit/AGM.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 --alpha 0.5
python train.py --config ./configs/AVE/release/vit/MLB.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 --alpha 3.0
python train.py --config ./configs/AVE/release/vit/OGM.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 --alpha 1.0
python train.py --config ./configs/AVE/release/vit/MMCosine.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8
python train.py --config ./configs/AVE/release/vit/MSLR.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8
python train.py --config ./configs/AVE/release/vit/ReconBoost.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha 3.0 --batch_size 8
python train.py --config ./configs/AVE/release/vit/MMPareto.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha 3.0 --batch_size 8
python train.py --config ./configs/AVE/release/vit/Reinit.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha 3.0 --kmepoch 3 --batch_size 8
python train.py --config ./configs/AVE/release/vit/MCR.json --default_config ./configs/AVE/default_config_ave_vit.json --fold 0 --lr 0.00001 --wd 0.0001 --l 0.01 --lib 1 --multil 1 --num_samples 32 --reg_by greedy --batch_size 8 --contr_coeff 1 --pre

####UCF
python train.py --config ./configs/UCF/res/unimodal_audio.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/unimodal_video.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/ens.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/joint_training.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/multiloss.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/pre_frozen.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/UCF/res/pre_finetuned.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.0001 --wd 0.0001
python train.py --config ./configs/UCF/res/MLB.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001 --alpha 2.0 --tanh_mode "2" --tanh_mode_beta 0.5
python train.py --config ./configs/UCF/res/MSLR.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/MMCosine.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001
python train.py --config ./configs/UCF/res/OGM.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001 --alpha 0.8
python train.py --config ./configs/UCF/res/AGM.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001 --alpha 2.0
python train.py --config ./configs/UCF/res/DnR.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001 --alpha 5.0 --kmepoch 3
python train.py --config ./configs/UCF/res/ReconBoost.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001 --alpha 0.5 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 4 --recon_ensemblestages 4
python train.py --config ./configs/UCF/res/MMPareto.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.001 --wd 0.0001 --alpha 3.0
python train.py --config ./configs/UCF/res/MCR.json --default_config   ./configs/UCF/default_config_ucf.json --fold 1 --lr 0.0001 --wd 0.0001 --l 0.001 --multil 0.1 --num_samples 32 --reg_by greedy --batch_size 32 --contr_coeff 1 --pre


##MOSEI V-T
python train.py --config ./configs/FactorCL/Mosei/release/VT/unimodal_video.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/unimodal_text.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/ens.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/joint_training.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/multiloss.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/pre_frozen.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/pre_finetuned.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/OGM.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --alpha 0.8 --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/AGM.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --alpha 0.5 --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/MLB.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --alpha 2.5 --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VT/MMPareto.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --alpha 3.0 --fold 0
python train.py --config ./configs/FactorCL/Mosei/release/VT/MCR.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VT.json --num_samples 32 --reg_by greedy --l 0.01 --multil 0.1 --contr_coeff 1  --lib 1 --fold 1 --pre


##MOSEI V-T-A
python train.py --config ./configs/FactorCL/Mosei/release/VTA/unimodal_video.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/unimodal_text.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/unimodal_audio.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/joint_training.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/multiloss.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/pre_frozen.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/pre_finetuned.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/ens.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/AGM.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --alpha 1.0 --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/MLB.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --alpha 0.5 --fold 1
python train.py --config ./configs/FactorCL/Mosei/release/VTA/MMPareto.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --alpha 1.0 --fold 0
python train.py --config ./configs/FactorCL/Mosei/release/VTA/MCR.json --default_config ./configs/FactorCL/Mosei/default_config_mosei_VTA.json --num_samples 32 --reg_by greedy --l 0.001 --multil 0.001 --contr_coeff 1 --pre --fold 0  --lib 1


##MOSI V-T
python train.py --config ./configs/FactorCL/Mosi/release/VT/unimodal_video.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/unimodal_text.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/ens.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --fold 0 --lr 0.0001 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/joint_training.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/multiloss.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --fold 0 --lr 0.0001 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/pre_frozen.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --fold 0 --lr 0.0001 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/pre_finetuned.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --fold 0 --lr 0.0001 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/OGM.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --alpha 0.8 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/AGM.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --alpha 1.5 --fold 0
python train.py --config ./configs/FactorCL/Mosi/release/VT/MLB.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --alpha 1.5 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/MMPareto.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --alpha 1.0 --fold 1 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VT/MCR.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json --lr 0.0001 --num_samples 32 --reg_by greedy --l 0.001 --multil 0.001 --contr_coeff 1 --fold 0


##MOSI V-T-A
python train.py --config ./configs/FactorCL/Mosi/release/VTA/unimodal_video.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/unimodal_text.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/unimodal_audio.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/ens.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/joint_training.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/multiloss.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/pre_frozen.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/pre_finetuned.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/AGM.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --alpha 0.5 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/MLB.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --alpha 1.5 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/MMPareto.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --alpha 1.5 --fold 0 --validate_with accuracy
python train.py --config ./configs/FactorCL/Mosi/release/VTA/MCR.json --default_config ./configs/FactorCL/Mosi/default_config_mosi_VTA.json --lr 0.0001 --num_samples 32 --reg_by greedy --l 0.01 --multil 0.01 --contr_coeff 1 --fold 0


##Sth-Sth
python train.py --config ./configs/SthSth/release/unimodal_video.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/unimodal_flow.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/ens.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/joint_training.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/multiloss.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/pre_frozen.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/pre_finetuned.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0
python train.py --config ./configs/SthSth/release/OGM.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0 --alpha 0.8
python train.py --config ./configs/SthSth/release/AGM.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0 --alpha 1.0
python train.py --config ./configs/SthSth/release/MLB.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0 --alpha 2.0 --tanh_mode_beta 0.5
python train.py --config ./configs/SthSth/release/ReconBoost.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0 --recon_weight1 5 --recon_weight2 1 --recon_epochstages 1 --recon_ensemblestages 1
python train.py --config ./configs/SthSth/release/MMPareto.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0 --alpha 3.0
python train.py --config ./configs/SthSth/release/DnR.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --fold 0 --alpha 3.0
python train.py --config ./configs/SthSth/release/MCR.json --default_config ./configs/SthSth/default_config_sthsth_2mod.json --lr 0.00001 --fold 0 --l 0.1 --multil 1 --num_samples 16 --reg_by greedy --contr_coeff 1 --lib 1 --pre



