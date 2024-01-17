import helper as hp
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--parameter", type=str, required=True, help='path to parameter.yml file')
    args = parser.parse_args()
    parameter_path = args.parameter
    
    # Read in parameter
    print()
    print('*** Running HMM guided mutagenesis script ***')
    PARAM = hp.read_yml_param_file(parameter_path)
    print('Read parameters: ########################')
    for k,v in PARAM.items():
        print(f'{k}: {v}')
    print('##########################################')
    print()
    
    is_verbose=PARAM['VERBOSE']
    
    # make directories to store outputs
    hp.make_directory(PARAM['FIG_DIR'], verbose=is_verbose)
    hp.make_directory(PARAM['DATA_DIR'], verbose=is_verbose)
    hp.myprint(verbose=is_verbose)
    
    # load trajectory data (pytraj.Trajectory = 0 indexing*)
    traj = hp.load_md_trajectory_and_parameter(PARAM['MD_TRAJECTORY_DATA_PATH'], PARAM['MD_PARAMETER_DATA_PATH'])
    topology = traj.top
    
    # Check whether test data length / sample timepoint length is appropriate
    hp.check_input_parameter(traj, PARAM, parameter_path)
    
    protein_res_ls = hp.store_residues_in_list(traj)
    aa_content = hp.get_aa_content(protein_res_ls)
    
    # Find amino acids found near substrate throughout md simulation
    print('*Extracting distance data from MD trajectory')
    atom_indices_multiple_frames = hp.find_atom_indices_near_sub(traj, PARAM)
    aa_near_sub = hp.find_aa_near_sub_from_indices(atom_indices_multiple_frames, topology, protein_res_ls, PARAM)
    hp.myprint(f"Number of amino acids found within {PARAM['SUB_AA_DIST']} of substate: ",len(aa_near_sub), verbose=is_verbose)
    hp.myprint(verbose=is_verbose)
    
    hp.save_aa_near_sub_data(aa_near_sub, 'aa_near_sub.txt', PARAM, verbose=is_verbose)
    hp.myprint(verbose=is_verbose)
    
    # Get substrate atom indices using pytraj
    substrate_atom_indicies = hp.get_sub_atom_indices(traj, PARAM)
    substrate_oxy_nit_indicies = hp.get_sub_o_n_atom_indices(traj, PARAM)
    hp.myprint('Substrate atom indices: ',substrate_atom_indicies, verbose=is_verbose)
    hp.myprint('Substrate oxygen & nitrogen indices: ',substrate_oxy_nit_indicies, verbose=is_verbose)
    hp.myprint(verbose=is_verbose)
    
    # Get alpha carbon indices of AA near substrate
    alpha_c_indices = hp.find_alpha_c_indices_in_aa_near_sub(traj, aa_near_sub)
    
    # Get hydrogen bond donors/acceptors (N&O) in AAs near substrate
    h_bond_donor_acceptor_indices = hp.find_h_bond_donors_acceptors_in_aa_near_sub(traj, aa_near_sub)
    
    # Get all pairs of (substrate atoms, alpha-carbons of aa near substrate) and (substrate atoms, alpha-carbons of aa near substrate)
    sub_aa_atom_pairs = [(x, y) for x in alpha_c_indices for y in substrate_atom_indicies] # sort & keep track of the alpha carbons for next step
    h_bond_pairs = [(x, y) for x in h_bond_donor_acceptor_indices for y in substrate_oxy_nit_indicies]
    pairs = sorted(sub_aa_atom_pairs) + h_bond_pairs
    hp.myprint('Number of alpha-carbon & substrate atom pairs: ',len(sub_aa_atom_pairs), verbose=is_verbose)
    hp.myprint('Number of h-bond donor & acceptor pairs: ',len(h_bond_pairs), verbose=is_verbose)
    hp.myprint('Total number of features (atom pairs): ',len(pairs), verbose=is_verbose)
    hp.myprint(verbose=is_verbose)
    
    hp.save_atom_pairs_data(pairs, 'pairs.txt', PARAM, verbose=is_verbose)
    
    # Get distance data of the extracted pairs
    data = hp.get_distance_data(traj, pairs)
    data = data.T
    
    hp.save_distance_data(data, 'distance_data.txt', PARAM, verbose=is_verbose)
    print('*Done')
    print()
    
    # Get indicator data
    if not PARAM['HAS_INDICATOR']:
        print('Conducting PCA to calculate RMSD indicator data')
        indicator_data_f_name='rmsd_indicator_data.txt'
        # If no reaction indicator pairs are provided, do coordinate PCA & cluster analysis to find stable frames (cluster centers)
        # Using center frame as reference, calculate rmsd & use it as indicator data
        projection_data, kde = hp.pca(traj, PARAM)
        cluster_center_frames = hp.find_center_of_cluster(projection_data.T, PARAM)
        
        # Choose cluster center frame randomly with weight
        centers_weights = hp.get_weights_of_cluster_centers(cluster_center_frames, kde)
        frames, rmsd_ls = hp.calc_rmsd_for_all_centers(traj, cluster_center_frames, PARAM)
        
        indicator_data = hp.get_rmsd_indicator_data(frames, rmsd_ls, centers_weights)    
    else:
        print('*Getting indicator data')
        indicator_data_f_name='indicator_data.txt'
        indicator_data = hp.calc_indicator_data(traj, PARAM['INDICATOR_PAIRS'])
    hp.save_indicator_data(indicator_data, indicator_data_f_name, PARAM, verbose=is_verbose)
    print('*Done')
    print()
        
    # Distance data preprocessing (dimension reduction) for HMM
    print('*Running data preprocessing')
    eigen_vals, eigen_vecs, data_standardized = hp.eigen_decomposition(data)
    n_pc = hp.select_nth_eigen_idx(eigen_vals, PARAM)
    hp.myprint(f"Reduced data from {data_standardized.shape[1]} into {n_pc} dimensions which covers {PARAM['VARIANCE_COVERAGE']*100}% of variance.", verbose=is_verbose)
    data_projected = hp.project_data_to_pc(data_standardized, eigen_vecs, n_pc)
    hp.myprint(verbose=is_verbose)
    
    # Split train/test data (test data selected evenly from beginning to end of total md simulation timepoints)
    test, train = hp.test_train_split(data_projected, PARAM)
    hp.myprint(f'Test Data Shape: {test.shape}, Train Data Shape: {train.shape}\n=> shape=(n_samples, n_timepoints, n_feature)', verbose=is_verbose)
    PARAM['N_TRAIN_SAMPLES']=len(train)
    
    # Needed to indicate seq timepoint length for each sample in training HMM
    train_lengths = [PARAM['SAMPLE_TP_LEN'] for _ in range(0,PARAM['N_TRAIN_SAMPLES'])] 
    reshaped_test=test.reshape(-1,test.shape[-1])
    reshaped_train=train.reshape(-1,train.shape[-1])
    print('*Done')
    print()
    
    if PARAM['DO_HYPERPARAM_SEARCH']:
        print('*Conducting hyperparamter search')
        print()
        lls_ls, aic_ls, bic_ls, best_n_components = hp.hyperparameter_search(reshaped_train, train_lengths, n_pc, PARAM)
        hp.plot_hyperparameter_search_output(lls_ls, aic_ls, bic_ls, PARAM)
        hp.myprint(f"Hyperparameter search plot (aic_bic.png) saved in {PARAM['FIG_DIR']}.", verbose=is_verbose)
    else:
        best_n_components=PARAM['HMM_N_COMPONENTS']
    
    # Use indicator data to score trajectory frames
    test_indicator_data = hp.slice_indicator_data_to_match_test_data(indicator_data, PARAM)
    
    # Run HMM
    print('*Running HMM...')
    high_sc_frames = hp.run_hmm_to_find_high_score_frames(data, reshaped_train, train_lengths, reshaped_test, test_indicator_data, \
                                                          best_n_components, PARAM)
    hp.myprint('Finished running HMM for %s iterations.' %PARAM['N_TEST_ITERATIONS'])
    print('*Done')
    print()
    
    print('*Seeking out trajectory frames with best score.')
    unique_frames = hp.get_unique_high_score_frames(high_sc_frames, PARAM)
    
    hp.save_high_sc_unique_frames_data(unique_frames, 'unique_frames.txt', PARAM, verbose=is_verbose)
    hp.myprint(f"Filtered trajectory frame indices data (unique_frames.txt) saved in {PARAM['DATA_DIR']}.", verbose=is_verbose)
    print('*Done')
    print()
    
    print('*Looking for residues for suggestion based on distance and residue/substrate_atom properties')
    # Slice original data using filtered frames
    filtered_frames_data = data[unique_frames,:len(sub_aa_atom_pairs)]
    col_mean, col_std = hp.get_mean_and_std_dev_of_filtered_data(filtered_frames_data)
    
    calpha_idx_to_resn_dict = hp.get_alphac_idx_to_resn_dict(traj, aa_near_sub) # output => 0 indexing
    
    res_idx_to_aa_sub_atom_data_dict = hp.get_res_idx_to_dist_data_dict(traj, PARAM, sub_aa_atom_pairs, \
                                                                     aa_near_sub, col_mean, col_std, \
                                                                     calpha_idx_to_resn_dict)
    aa_near_sub_resi_ls, aa_near_sub_resn_ls, \
    aa_near_sub_top3_atoms_ls, \
    aa_near_sub_aa_suggestion_and_length_ls = hp.loop_sub_near_aa_to_suggest_aa(traj, PARAM, res_idx_to_aa_sub_atom_data_dict)
    print('*Done')
    print()
    
    print('*Generating final output')
    resi_to_aa_suggestions_dict = hp.get_final_output(aa_near_sub_resi_ls, \
                                                   aa_near_sub_resn_ls, \
                                                   aa_near_sub_top3_atoms_ls, \
                                                   aa_near_sub_aa_suggestion_and_length_ls)
    
    hp.save_final_out_as_csv(resi_to_aa_suggestions_dict, 'final_out.csv', PARAM, verbose=is_verbose)
    print('*Done')
    print()
    print("Script Finished. Exiting.")
    
if __name__ == "__main__":
    main()
    
    
    
    
    