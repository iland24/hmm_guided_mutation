import os
import re
import yaml
import pytraj as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# Constants ################################################################

# length of AA when stretched out vertically (from: BioNumbers database)
AA_LEN = {'ALA': 5.5, 'ARG': 11.0, 'ASN': 7.5, 'ASP': 6.5,
               'CYS': 6.4, 'GLU': 8.0, 'GLN': 9.0, 'GLY': 3.9,  
               'HIS': 8.5, 'ILE': 8.5, 'LEU': 8.5, 'LYS': 11.3,
               'MET': 10.3, 'PHE': 9.7, 'PRO': 6.2, 'SER': 6.1,
               'THR': 6.1, 'TRP': 10.9, 'TYR': 10.4, 'VAL': 7.0}

# Three letter AA codes (includes AMBER Residue names of Common Non-standard Protonation States)
AA = ['ALA','ARG','ASN','ASP','ASH','CYS','CYM','CYX','GLU','GLH',
      'GLN','GLY','HIS','HID','HIE','HIP','ILE','LEU','LYS','LYN',
      'MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

PROPERTY_AA={'polar':['THR','SER','ASN','GLN','CYS','TYR'],
             'nonpolar':['ALA','GLY','ILE','LEU','MET','TRP','PHE','VAL','PRO'],
             'positive':['HIS','LYS','ARG'],
             'negative':['GLU','ASP']}

# Verbose printing util function #####################################

def myprint(*mystr, verbose=True, end='\n'):
    if verbose:
        print(*mystr,end=end)
    
# parameter/data loading functions ###########################################

def read_yml_param_file(filepath):
    with open(filepath,'r') as f:
        param = yaml.safe_load(f)
    return param

def load_md_trajectory_and_parameter(traj_f_name, parm_f_name):
    """
    loads trajectory and parameter data in cwd
    
    returns pytraj trajectory object
    """
    if os.path.exists(traj_f_name) and os.path.exists(parm_f_name):
        traj = pt.load(traj_f_name, top=parm_f_name)
        orig_len=len(traj)
        # Drop all frames below biggest place-value of total frame number
        n_digit = len(str(abs(len(traj))))-1
        max_frame = (len(traj)//10**(n_digit))*10**n_digit
        traj=traj[:max_frame]
        new_len=len(traj)
        if orig_len != new_len:
            print(f"Traj frames reduced from {orig_len} to {new_len}.")
        print(f"Trajectory data loaded: {new_len} frames.", end='\n\n')
        return traj
    else:
        print(f"Either one or both \'{traj_f_name}\' and/or \'{parm_f_name}\' not found in given path.")
        print('Exiting program')
        exit()

def check_input_parameter(traj, PARAM, parameter_path):
    is_ok1 = check_sample_tp_len(traj, PARAM, parameter_path)
    is_ok2 = check_total_test_tp_len(traj, PARAM, parameter_path)
    n_train_samples=len(traj)//PARAM['SAMPLE_TP_LEN'] - PARAM['N_TEST_SAMPLES']
    if is_ok1 and is_ok2:
        print(f"Proceding with given SAMPLE_TP_LEN ({PARAM['SAMPLE_TP_LEN']}), N_TEST_SAMPLES ({n_train_samples}) and N_TEST_SAMPLES ({PARAM['N_TEST_SAMPLES']})")
        print()
        
def check_sample_tp_len(traj, PARAM, parameter_path):
    # Each sample tp len should not be bigger than len(traj)    
    sample_tp_len = PARAM['SAMPLE_TP_LEN']
    if sample_tp_len > len(traj):
        print(f"Each sample\'s timepoints (=>{sample_tp_len}) is longer than total trajectory data timepoints(=>{len(traj)}).\n"
              f"Please change \'SAMPLE_TP__LEN\' value in parameter file at {parameter_path}.")
        return False
    return True

def check_total_test_tp_len(traj, PARAM, parameter_path):
    #  Not advised for total test data timepoints to be more than 50% of len(traj)
    tot_test_tp_len = PARAM['SAMPLE_TP_LEN']*PARAM['N_TEST_SAMPLES']
    percent_test = int(round(tot_test_tp_len/len(traj),2)*100)
    if percent_test > 50:
        print(f"Test data timepoints is longer than training data timepoints. Test data: {percent_test}% of total traj timepoints/frames.\n",
              f"Please change \'SAMPLE_TP__LEN\' and/or \'N_TEST_SAMPLES\' value in parameter file at {parameter_path}.")
        return False
    return True

# Data saving functions ##############################################

def make_directory(dir_name, verbose):
    """
    creates a directory inside current dir
    with name dir_name
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        myprint(f"Folder \'{dir_name}\' created!", verbose=verbose)
    else:
        myprint(f"Folder \'{dir_name}\' already exists. Directory not created.", verbose=verbose)

def get_f_name_from_path(path):
    split_path = os.path.splitext(path)
    if '/' in split_path[0]:
        return re.search('[0-9a-zA-Z-_@#$%&]*$', split_path[0]).group()
    else:
        return split_path[0]

def make_unique_file_name(path, PARAM):
    filename, extension = os.path.splitext(path)
    counter = 1
    path = filename + "_" + get_f_name_from_path(PARAM['MD_TRAJECTORY_DATA_PATH']) + "_" + str(counter) + extension
    while os.path.exists(path):
        path = filename + "_" + get_f_name_from_path(PARAM['MD_TRAJECTORY_DATA_PATH']) +"_" + str(counter) + extension
        counter += 1
    return path

def save_aa_near_sub_data(aa_near_sub, f_name, PARAM, verbose):
    path=PARAM['DATA_DIR']
    p = make_unique_file_name(path+'/'+f_name, PARAM)
    if not os.path.exists(p):
        with open(p,'w') as f:
            for aa in aa_near_sub:
                f.write(f'{aa[0]} {aa[1]}\n')
        myprint(f"File \'{p}\' created!", verbose=verbose)
    else:
        myprint(f"File \'{p}\' already exists. Saving with an alternative name.", verbose=verbose)
        new_path = make_unique_file_name(p, PARAM)
        with open(new_path,'w') as f:
            for aa in aa_near_sub:
                f.write(f'{aa[0]} {aa[1]}\n')
        myprint(f"File \'{new_path}\' created!", verbose=verbose)

def save_atom_pairs_data(pairs, f_name, PARAM, verbose):
    path=PARAM['DATA_DIR']
    p = make_unique_file_name(path+'/'+f_name, PARAM)
    if not os.path.exists(p):
        with open(p,'w') as f:
            for pair in pairs:
                f.write(f'{pair[0]} {pair[1]}\n')
        myprint(f"File \'{p}\' created!", verbose=verbose)
    else:
        myprint(f"File \'{p}\' already exists. Saving with an alternative name.", verbose=verbose)
        new_path = make_unique_file_name(p, PARAM)
        with open(new_path,'w') as f:
            for pair in pairs:
                f.write(f'{pair[0]} {pair[1]}\n')
        myprint(f"File \'{new_path}\' created!", verbose=verbose)

def save_distance_data(data, f_name, PARAM, verbose):
    path=PARAM['DATA_DIR']
    p = make_unique_file_name(path+'/'+f_name, PARAM)
    if not os.path.exists(p):
        np.savetxt(p, data, delimiter=',')
        myprint(f"File \'{p}\' created!", verbose=verbose)
    else:
        myprint(f"File \'{p}\' already exists. Saving with an alternative name.", verbose=verbose)
        new_path = make_unique_file_name(p, PARAM)
        np.savetxt(new_path, data, delimiter=',')
        myprint(f"File \'{new_path}\' created!", verbose=verbose)

def save_indicator_data(indicator_data, f_name, PARAM, verbose):
    path=PARAM['DATA_DIR']
    p = make_unique_file_name(path+'/'+f_name, PARAM)
    if not os.path.exists(p):
        np.savetxt(p, indicator_data, delimiter=',')
        myprint(f"File \'{p}\' created!", verbose=verbose)
    else:
        myprint(f"File \'{p}\' already exists. Saving with an alternative name.", verbose=verbose)
        new_path = make_unique_file_name(p, PARAM)
        np.savetxt(new_path, indicator_data, delimiter=',')
        myprint(f"File \'{new_path}\' created!", verbose=verbose)

def save_high_sc_unique_frames_data(unique_frames, f_name, PARAM, verbose):
    path=PARAM['DATA_DIR']
    p = make_unique_file_name(path+'/'+f_name, PARAM)
    if not os.path.exists(p):
        with open(p,'w') as f:
            for frame in unique_frames:
                f.write(f'{frame}\n')
        myprint(f"File {p} created!", verbose=verbose)
    else:
        myprint(f"File {p} already exists. Saving with an alternative name.", verbose=verbose)
        new_path = make_unique_file_name(p, PARAM)
        with open(new_path,'w') as f:
            for frame in unique_frames:
                f.write(f'{frame}\n')
        myprint(f"File {new_path} created!", verbose=verbose)

def save_final_out_as_csv(resi_to_aa_suggestions_dict, f_name, PARAM, verbose):
    path=PARAM['DATA_DIR']
    p = make_unique_file_name(path+'/'+f_name, PARAM)
    if not os.path.exists(p):
        final_out = pd.DataFrame(resi_to_aa_suggestions_dict, index=['Current Residue','nearby 3 substrate atoms','AA Suggestion and Length'])
        final_out.to_csv(p)
        myprint(f"File \'{p}\' created!", verbose=verbose)
    else:
        myprint(f"File \'{p}\' already exists. Saving with an alternative name.", verbose=verbose)
        new_path = make_unique_file_name(p, PARAM)
        final_out = pd.DataFrame(resi_to_aa_suggestions_dict, index=['Current Residue','nearby 3 substrate atoms','AA Suggestion and Length'])
        final_out.to_csv(new_path)
        myprint(f"File \'{new_path}\' created!", verbose=verbose)
        
# Data/Feature extraction functions ############################################

def store_residues_in_list(traj):
    """
    returns list containing sequence of 
    residue names(capitalized three letter code)
    of enzyme used in MD
    """
    topology = traj.top
    protein_res_ls=[]
    # name: residue name, index: residue index
    for v in topology.residues:
        protein_res_ls.append(v.name)
    return protein_res_ls

def get_original_aa_name(aa):
    """
    checks given capitalized three letter
    amino acid code is an alternative version
    of protonated state name
    
    returns original cap three letter code of amino acid
    """
    if aa in ['HID','HIE','HIP']:
        return'HIS'
    elif aa in ['CYM','CYX']:
        return 'CYS'
    elif aa=='ASH':
        return 'ASP'
    elif aa=='GLH':
        return 'GLU'
    elif aa=='LYN':
        return 'LYS'
    else:
        return aa
        
def get_aa_content(protein_res_ls):
    """
    returns percentage of residue content in 
    protein currently being analyzed as pd.Series
    """
    # get residue content in % of our protein
    for i,aa in enumerate(protein_res_ls):
        protein_res_ls[i] = get_original_aa_name(aa)
            
    aa_seq = [aa for aa in protein_res_ls if aa in AA]
    aa_content = pd.Series(aa_seq).value_counts()/len(aa_seq)*100
    
    return aa_content

def get_sub_atom_indices(traj, PARAM):
    """
    returns list of indices of carbon, oxygen
    and nitrogen atoms in substrate
    """
    return list(traj.top.atom_indices(':'+PARAM['SUBSTRATE_NAME']+'@C*,O*,N*'))

def get_sub_o_n_atom_indices(traj, PARAM):
    """
    returns list of indices of oxygen
    and nitrogen atoms in substrate
    """
    return list(traj.top.atom_indices(':'+PARAM['SUBSTRATE_NAME']+'@O*,N*'))

def find_atom_indices_near_sub(traj, PARAM):
    """
    return list of atom indices near substrate 
    using downsampled traj data
    """
    downsampled_traj = traj[::PARAM['TRAJ_DOWNSAMPLE_FACTOR']]
    sub_aa_dist=PARAM['SUB_AA_DIST']
    sub_name=PARAM['SUBSTRATE_NAME']
    
    # atoms within 4 ang of SUB residue
    atom_indices_multiple_frames = pt.search_neighbors(downsampled_traj, f':{sub_name}<@{sub_aa_dist}')
    
    return atom_indices_multiple_frames

def find_aa_near_sub_from_indices(atom_indices_multiple_frames, topology, protein_res_ls, PARAM):
    """
    return list of amino acids near substrate
    
    *each element of the list is a tuple(residue index, residue name)
    """
    res_multiple_frames=[]
    for atom_indices_one_frame in atom_indices_multiple_frames:
        res_one_frame=[]
        for atom in atom_indices_one_frame:
            # add 1 to resid to make residue index 1 indexing
            res_one_frame.append((topology[atom].resid+1, protein_res_ls[topology[atom].resid] )) 
        res_one_frame=list(set(res_one_frame))
        res_multiple_frames.append(res_one_frame)
    
    aa_near_sub=[]
    for res_one_frame in res_multiple_frames:
        for res in res_one_frame:
            if res not in aa_near_sub:
                aa_near_sub.append(res)
    
    # remove non-aa
    aa_near_sub = [aa for aa in aa_near_sub if aa[1] in AA]
    
    # convert alternative three letter aa code to original aa code
    for i,aa in enumerate(aa_near_sub):
        aa_near_sub[i] = get_original_aa_name(aa)
        
    aa_near_sub = sorted(aa_near_sub,key=lambda x:x[0])
    
    return aa_near_sub

def aa_content_of_aa_near_sub(aa_near_sub):
    """
    returns percentage of amino acids
    content of the amino acids found near
    substrate as Pandas Series
    
    *series index: amino acid
    *series value: content in percentage
    """
    return pd.Series([aa[1] for aa in aa_near_sub]).value_counts()/len(aa_near_sub)*100

def find_alpha_c_indices_in_aa_near_sub(traj, aa_near_sub):
    """
    returns list of alpha carbon indices 
    of amino acids found near substrate
    """
    alpha_c_indices=[]
    for res in aa_near_sub:
         # cpptraj masking => require 1 indexing*
        alpha_c_indices=alpha_c_indices+list(pt.select(f':{res[0]}@CA',traj.top))
        
    return alpha_c_indices

def find_h_bond_donors_acceptors_in_aa_near_sub(traj, aa_near_sub):
    """
    returns list of potential hydrogen bond donor indices 
    of amino acids found near substrate
    """
    h_bond_donor_acceptor_indices=[]
    for res in aa_near_sub:
        # ? in cpptraj mask below enables choosing non-backbone oxygens and nitrogens
        # cpptraj masking => require 1 indexing*
        result=pt.select(f':{res[0]}@O?*,N?*',traj.top) 
        if len(result)!=0:
            h_bond_donor_acceptor_indices=h_bond_donor_acceptor_indices+list(result)
            
    return h_bond_donor_acceptor_indices

def get_distance_data(traj,pairs):
    """
    return distance from all the atom pairs
    """
    return pt.distance(traj, pairs)

def get_alphac_idx_resname_dict(traj, aa_near_sub):
    """
    returns dictionary with alpha carbon
    index of amino acids near substrate as 
    key and residue name as value
    """
    calpha_idx_resname_dict={}
    for res in aa_near_sub:
        calpha_idx_resname_dict[pt.select(f':{res[0]}@CA',traj.top)[0]]=res[1]
        
    return calpha_idx_resname_dict

def aa_content_of_aa_near_sub(aa_near_sub):
    """
    returns percentage of amino acids
    content of the amino acids found near
    substrate as Pandas Series
    
    *series index: amino acid
    *series value: content in percentage
    """
    return pd.Series([aa[1] for aa in aa_near_sub]).value_counts()/len(aa_near_sub)*100

# PCA & calculating RMSD indicator data functions ###################################
# (when indicator data is not given)

def save_pca_fig(projection_data, kde, PARAM):
    """
    projected data: data projected to PC1 and 2
    kde: kernel density estimation of projected data
    path: path to figure directory
    
    saves pca plot to path
    """
    plt.scatter(projection_data[0], projection_data[1], marker='o', alpha=0.5, c=kde)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    cbar = plt.colorbar()
    cbar.set_label('kde')
    unique_path = make_unique_file_name(PARAM['FIG_DIR']+'/pca_coordinate_data.png', PARAM)
    plt.savefig(unique_path)
    plt.close()

def pca(traj, PARAM):
    """
    Perform PCA analysis on trajectory
    coordinate data of nitrogen, oxygen and alpha-carbon
    
    return projected data and kernel 
    density estimation data of the projected data
    """
    pca = PCA(n_components=2)
    traj_new = traj[0::PARAM['PCA_DOWNSAMPLE_FACTOR']]
    traj_new = traj_new['@O,N,CA']
    
    xyz_2d = traj_new.xyz.reshape(traj_new.n_frames, traj_new.n_atoms * 3)
    # print(xyz_2d.shape)  # (n_frames, n_dimensions)
    reduced_cartesian = pca.fit_transform(xyz_2d)
    reduced_cartesian=reduced_cartesian.T
    
    z = gaussian_kde(reduced_cartesian)(reduced_cartesian) # calc estimated density for projected points
    save_pca_fig(reduced_cartesian, z, PARAM)
    
    return reduced_cartesian, z
def find_closest_vector(target_vector, vectors):
    """
    returns a vector closest (Euclidian distance) 
    to target vector from vectors
    """
    # Calculate Euclidean distances
    distances = np.linalg.norm(vectors - target_vector, axis=1)
    # Find the index of the vector with the minimum distance
    closest_index = np.argmin(distances)
    # Return the closest vector & its index
    closest_vector = vectors[closest_index]
    
    return (closest_index, closest_vector)

def get_elbow_idx(inertia_ls):
    """
    Estimate the elbow point of the inertia plot
    based on variance coverage (set as 0.8) of 
    inertia data.
    """
    inertia_tot = sum(inertia_ls)
    inertia_sum = 0
    over_eighty_coverage_idx = None
    for i,val in enumerate(inertia_ls):
        inertia_sum+=val
        if inertia_sum/inertia_tot > 0.8:
            over_eighty_coverage_idx = i
            break
    sliced_inertia_ls = inertia_ls[over_eighty_coverage_idx+1:]
    mean = np.mean(sliced_inertia_ls)
    std = np.std(sliced_inertia_ls)
    upper_bound = mean+std
    
    new_inertia_ls=[]
    for val in inertia_ls:
        if val>upper_bound:
            new_inertia_ls.append(val)
    return len(new_inertia_ls)
    
def find_center_of_cluster(data, PARAM):
    """
    data: PCA analysis output
    shape=(n_frames, n_PC)
    
    Performs k-means to find clusters in PCA.
    
    Returns frames closest to the centers.
    
    *cluster_center_frames
    = tuple(frame number,coordinate)
    """
    
    k_vals = range(1, 10)

    # Run k-means for each k and store the inertia in a list
    inertia = []
    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.plot(k_vals, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')

    plt.savefig(PARAM['FIG_DIR']+'/k_mean_inertia.png')
    plt.close()
    
    # get index of elbow from elbow method
    optimal_k = get_elbow_idx(inertia)
    
    kmeans = KMeans(n_clusters = optimal_k).fit(data)
    cluster_center_frames=[]
    for cluster_center in kmeans.cluster_centers_:
        target_vec = cluster_center
        vecs = data[:]
        idx,vec = find_closest_vector(target_vec,vecs)
        cluster_center_frames.append((idx,vec))

    return cluster_center_frames

def get_weights_of_cluster_centers(cluster_center_frames, kde):
    """
    returns calculated weight of each center 
    based on kernel density estimation
    """
    center_weight=np.zeros((len(cluster_center_frames)))
    density_sum=0
    for i,center in enumerate(cluster_center_frames):
        center_weight[i] = kde[center[0]]
        density_sum += kde[center[0]]
        
    return center_weight/density_sum

def calc_rmsd_for_all_centers(traj, cluster_center_frames, PARAM):
    """
    returns root mean squared deviation for all timepoints
    using cluster center as the reference frame
    """
    frames=[x[0] for x in cluster_center_frames]
    rmsd_ls=[]
    for frame in frames:
        # multiply by downsample factor to get frame idx in original traj data
        rmsd_ls.append(pt.rmsd(traj, ref=traj[frame*PARAM['PCA_DOWNSAMPLE_FACTOR'] ], mask='@O,N,C,CA')) 
        
    return frames, rmsd_ls

def get_rmsd_indicator_data(frames, rmsd_ls, centers_weights):
    """
    returns rmsd indicator data chosen randomly with weights 
    based on pca & k-means outputs
    """
    frame_choice = np.random.choice(frames, p=centers_weights)
    
    return rmsd_ls[frames.index(frame_choice)]


# Calculating indicator data functions ####################################################

def calc_indicator_data(traj, pairs):
    """
    Calculates indicator distance data based on
    given atom pairs
    
    returns indicator data
    shape=(total timepoints,1)
    """
    # Get reaction indicator distances
    indicator_data = pt.distance(traj, pairs)
    # If there are several reaction indicator atom pairs, sum them up
    indicator_data = np.sum(indicator_data,axis=0)
    indicator_data = indicator_data.reshape(-1,1)
    
    return indicator_data

def slice_indicator_data_to_match_test_data(indicator_data, PARAM):
    """    
    Indicator data contains data for all MD timepoints.
    To match the test data, portion for train data must be sliced out.

    indicator_data: distance data used to indicate
    whether the reaction is more or less likely to occur.
    shape=(total_MD_timepoints,1)

    returns indicator data that matches the test data.
    shape=(PARAM['N_TEST_SAMPLES'], PARAM['SAMPLE_TP_LEN'])
    """
    sample_tp_len = PARAM['SAMPLE_TP_LEN']
    n_test = PARAM['N_TEST_SAMPLES']
    
    dim1 = int(indicator_data.shape[0]/sample_tp_len)
    dim2 = sample_tp_len
    indic_data_reshaped = indicator_data.reshape((dim1,dim2))
    
    interval = max(1, len(indic_data_reshaped) // n_test) 
    indic_data_mask = list(range(0, len(indic_data_reshaped), interval))
    test_indicator_data = indic_data_reshaped[indic_data_mask]
    return test_indicator_data


# Preprocessing distance data for HMM functions ###############################

def eigen_decomposition(data):
    """
    returns eigen values, eigen vectors 
    and standardized data
    """
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    data_cov=np.cov(data_std, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eig(data_cov)
    return eigen_vals, eigen_vecs, data_std

def select_nth_eigen_idx(eigen_vals, PARAM):
    """
    returns index of eigen value whose sum 
    upto that eigen value covers PARAM['VARIANCE_COVERAGE'] of variance
    """
    tot_var=0
    n_pc=0
    for i,ev in enumerate(eigen_vals):
        tot_var+=ev/sum(eigen_vals)
        if tot_var>=PARAM['VARIANCE_COVERAGE']:
            n_pc=i
            break
            
    return n_pc

def project_data_to_pc(data_standardized, eigen_vecs, n_pc):
    """
    return data projected to given 
    number of principal components
    """
    return np.dot(data_standardized, eigen_vecs[:,:n_pc])

def test_train_split(data, PARAM):
    """
    data: dim reduced data; shape:(total_md_tp, n_features)
    
    Returns split test and train data.
    
    Test data is collected by sampling at 
    an even interval based on number of test data.
    """
    sample_tp_len=PARAM['SAMPLE_TP_LEN']
    tot_tp=data.shape[0]
    n_feat=data.shape[1]

    dim1=int(tot_tp/sample_tp_len)
    dim2=sample_tp_len
    
    # reshape data => shape=(n_samples, n_timepoints_in_sample, n_features)
    data_reshaped = data.reshape((dim1,dim2,-1))
    
    # Choose test samples evenly throughout total MD timepoints
    if PARAM['N_TEST_SAMPLES'] > len(data_reshaped):
        print(('Number of test samples should not be greater than number'
               'of total data samples.\nEither reduce the N_TEST_SAMPLES ' 
               'or reduce the N_SAMPLE_TIMEPOINTS'), end='\n\n')
        exit()
        print('Exiting program')
    
    interval = len(data_reshaped) // PARAM['N_TEST_SAMPLES']
    test_mask = list(range(0, len(data_reshaped), interval))
    test_data = data_reshaped[test_mask]
    # Choose train samples (!test samples)
    train_mask=np.ones(len(data_reshaped),dtype=bool)
    train_mask[test_mask]=False
    train_data = data_reshaped[train_mask]
    
    return test_data, train_data

# Hyperparameter search functions ########################################

def get_n_params(nc,nf):
    """
    nc: number of components/states
    nf: number of features

    returns total number of parameters of HMM model
    based on nc and nf
    """
    return sum({
        "s": nc - 1,
        "t": nc * (nc - 1),
        "m": nc * nf,
        "c": nc * nf,
    }.values())

def hyperparameter_search(train_data, train_lengths, n_pc, PARAM):
    """
    fits gaussian hmm models using list of n_components
    and calculates loglikelihood, aic and bic.
    
    returns list of loglikelhood, aic, bic, and best n_component
    """
    # Hyperparameter Optimization
    best_n_components = None
    lls_ls = []
    aic_ls = []
    bic_ls = []

    n_features = n_pc # from dim reduction step

    for n_components in PARAM['HMM_N_COMPONENTS_LS']:
        best_score = None
        best_model = None
        for _ in range(10):
            # Create and fit the model
            model = hmm.GaussianHMM(n_components=n_components, n_iter=PARAM['HMM_TRAIN_ITERATIONS'])
            model.fit(train_data,train_lengths)

            # Evaluate model (replace with your own evaluation metric)
            score = model.score(train_data,train_lengths)

            # Update best parameters if needed
            if not best_score or best_score < score:
                best_score = score
                best_n_components = n_components
                best_model = model

        lls_ls.append(best_model.score(train_data))
        
        # Use best model's loglikelihood to calc aic & bic
        n_params = get_n_params(n_components,n_features)
        aic_ls.append(-2 * best_model.score(train_data, train_lengths) + 2 * n_params)
        bic_ls.append(-2 * best_model.score(train_data, train_lengths) + n_params * np.log(len(train_data)))
    
    print("*Hyperparameter output*")
    myprint("Best LogLikelihood: ", best_score, verbose=PARAM['VERBOSE'])
    print('Selected number of HMM states: ', best_n_components)
    print()
    
    # Choose index of metric list that maximize the condition (maximize lls but minimize aic&bic)
    combined_metric = [ll - aic - bic for ll, aic, bic in zip(lls_ls, aic_ls, bic_ls)]
    max_idx = max(range(len(combined_metric)), key=combined_metric.__getitem__)

    # Choose best number of components
    best_n_components = PARAM['HMM_N_COMPONENTS_LS'][max_idx]
    
    return lls_ls, aic_ls, bic_ls, best_n_components

def plot_hyperparameter_search_output(lls_ls, aic_ls, bic_ls, PARAM):
    # plot & save  
    ns=PARAM['HMM_N_COMPONENTS_LS']

    fig, ax = plt.subplots()
    ln1 = ax.plot(ns, aic_ls, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic_ls, label="BIC", color="green", marker="o")
    ax2 = ax.twinx() # = plot lls on same as axis as aic & bic
    ln3 = ax2.plot(ns, lls_ls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines,loc='center right')
    ax.set_title("Using AIC/BIC for Model Selection")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    fig.tight_layout()
    unique_path = make_unique_file_name(PARAM['FIG_DIR']+'/aic_bic.png', PARAM)
    plt.savefig(unique_path)
    plt.close()

# HMM & state scoring functions #################################################

def get_state_score(indicator_data, state_seq, n_components):
    """
    indicator_data: distance data used to indicate
    whether the reaction is more or less likely to occur.
    shape=(PARAM['N_TEST_SAMPLES'], PARAM['SAMPLE_TP_LEN'])
    
    state_seq: viterbi state path sequence data.
    shape=(PARAM['N_TEST_SAMPLES'], PARAM['SAMPLE_TP_LEN'])
    *hmm_learn states starts from 0
    
    returns np.array of score of each hidden state
    shape=(n_components)
    """
    state_scores=np.zeros((n_components))
    state_cnt=np.zeros((n_components))

    for indic_i, state_i in zip(indicator_data,state_seq): # loop sample
        for indic_j,state_j in zip(indic_i,state_i): # loop tp
            state_scores[int(state_j)]+=indic_j
            state_cnt[int(state_j)]+=1
    
    # account for cases when there is 0 cnt
    for i in range(len(state_scores)):
        if state_cnt[i]==0: 
            state_scores[i]=float('inf')
        else:
            state_scores[i]=state_scores[i]/state_cnt[i]
            
    return state_scores

def get_high_score_frames(data, test_state_seq, state_score_arr, PARAM):
    """
    returns list of frame indices of original trajectory data
    labeled with best scoring state
    """
    best_state=np.argmin(state_score_arr)
    
    tot_tp=data.shape[0]
    interval = int(tot_tp/PARAM['SAMPLE_TP_LEN']) // PARAM['N_TEST_SAMPLES']
    
    frames=[]
    for s_idx, sample in enumerate(test_state_seq):
        best_state_indices = np.where(sample==best_state)[0]
        for idx in best_state_indices:
            # convert test frame index to original frame index
            frames.append(s_idx*interval*100+idx)
            
    return frames

def run_hmm_to_find_high_score_frames(data, train_data, train_lengths, test_data, test_indicator_data, best_n_components, PARAM):
    """
    Train HMM and use viterbi decoder 
    on test data PARAM['N_TEST_ITERATIONS'] times.
    
    Return filtered frames with highest scoring state 
    based on indicator data.
    
    *shape/length of filtered frames vary based on number
    of frames labled with highest scoring state
    """
    # Run HMM using best_n_compo
    high_sc_frames=[]
    for _ in range(PARAM['N_TEST_ITERATIONS']):
        # Using best hyperparam, fit hmm and get viterbi paths of test data
        model = hmm.GaussianHMM(n_components=best_n_components, n_iter=PARAM['HMM_TRAIN_ITERATIONS'])
        model.fit(train_data, train_lengths)

        # get state sequences for test data
        test_lengths = [PARAM['SAMPLE_TP_LEN'] for _ in range(0,PARAM['N_TEST_SAMPLES'])]
        _, state_seq = model.decode(test_data, lengths=test_lengths, algorithm='viterbi')
        test_state_seq = state_seq.reshape((PARAM['N_TEST_SAMPLES'], PARAM['SAMPLE_TP_LEN']))
        
        # score each state
        state_score_arr = get_state_score(test_indicator_data, test_state_seq, best_n_components)
        frames = get_high_score_frames(data, test_state_seq, state_score_arr, PARAM)
        high_sc_frames.append(frames)
        
    return high_sc_frames

def get_unique_high_score_frames(high_sc_frames, PARAM):
    """
    return unique frames that appear more than 50% 
    than PARAM['N_TEST_ITERATIONS'] from list of high scoring frames
    """
    flat_high_sc_frames=[]
    for trial in high_sc_frames:
        flat_high_sc_frames+=list(trial)

    # Flatten frame index data
    flat_high_sc_frames=np.array(flat_high_sc_frames)

    # Get Counts of unique frames
    unique_frames, counts = np.unique(flat_high_sc_frames, return_counts=True)

    # Discard frames that are seen less than 50% of total number of trials
    unique_frames = unique_frames[np.where(counts > int(PARAM['N_TEST_ITERATIONS']*.5))[0]]
    
    return unique_frames

def get_mean_and_std_dev_of_filtered_data(filtered_frames_data):
    """
    return means and standard deviations 
    of all columns of data with filtered frames
    """
    mean=np.mean(filtered_frames_data,axis=0)
    std=np.std(filtered_frames_data,axis=0)
    return mean,std

def get_alphac_idx_to_resn_dict(traj, aa_near_sub):
    """
    loops through aa_near_sub (=> list of (aa_idx, aa_name))
    
    returns dictionary with alpha carbon
    index as key and (res_name,res_idx) as value.
    
    *returned res_idx is 0 indexing
    """
    calpha_idx_to_resn_dict={}
    for res in aa_near_sub:
        # cpptraj mask => 1 indexing
        calpha_idx_to_resn_dict[pt.select(f':{res[0]}@CA',traj.top)[0]]=(res[1],res[0]-1)
        
    return calpha_idx_to_resn_dict # output => 0 indexing

def get_aa_property(aa):
    """
    aa: capitalized three letter aa code
    
    returns aa proterty label
    *property labels: polar, nonpolar, positive, negative
    """
    if aa in PROPERTY_AA['polar']:
        return 'polar'
    elif aa in PROPERTY_AA['nonpolar']:
        return 'nonpolar'
    elif aa in PROPERTY_AA['positive']:
        return 'positive'
    elif aa in PROPERTY_AA['negative']:
        return 'negative'
    else:
        print(f'Amino acid code not found: {aa}')

def get_sub_atom_idx_to_name_dict(traj, PARAM):
    """
    return substrate atom index to name dict 
    (atom idx: 0 indexing)
    """
    topology=traj.top
    sub_atom_idx_to_name_dict={}
    sub_atoms_indices=list(traj.top.atom_indices(':'+PARAM['SUBSTRATE_NAME']+'@O*,N*,C*'))
    
    for a_idx in sub_atoms_indices:
        sub_atom_idx_to_name_dict[a_idx]=topology.atom(a_idx).name[0]
    
    return sub_atom_idx_to_name_dict

def get_res_idx_to_dist_data_dict(traj, PARAM, sub_aa_atom_pairs, aa_near_sub, col_mean, col_std, calpha_idx_to_resn_dict):
    """
    return residue index & name to distance & substrate 
    (key: tuple(resi,resn), value: [mean, standard deviation, atom index])
    """
    res_idx_to_aa_sub_atom_data_dict={}
    
    sub_atom_idx_to_name_dict = get_sub_atom_idx_to_name_dict(traj, PARAM)
    # get keys(resn) from aa_near_sub first 
    for aa in aa_near_sub:
        res_idx_to_aa_sub_atom_data_dict[aa[0]-1]=[] # change resn idx to 0 indexing
    
    # store data list using resi as key
    resi_to_dist_data_dict = {}
    for m,s,pair in zip(col_mean, col_std, sub_aa_atom_pairs): # sub_aa_atom_pairs => 0 indexing
        resi = calpha_idx_to_resn_dict[pair[0]][1]
        resi_to_dist_data_dict[resi]=\
        resi_to_dist_data_dict.get(resi,[])+[(calpha_idx_to_resn_dict[pair[0]][0], round(m,2), round(s,2), pair[1])]

    # sort based on mean distance & slice to get top3 shortest mean distances
    for k,v in resi_to_dist_data_dict.items():
        resi_to_dist_data_dict[k]= sorted(v,key=lambda x:x[1])[:3]

    return resi_to_dist_data_dict

def is_aa_within_range(res_name, mean, std, n_std):
    """
    Returns True if given aa is within the bound 
    of 3 standard deviation away from mean, else
    False
    
    n_std: 1 or 2 or 3
    find data within n_std away from given mean
    """
    upperbound=mean+std*n_std
    lowerbound=mean-std*n_std
    
    # print(res_name,AA_LEN[res_name], ':',get_aa_property(res_name))
    # print('upper: ',upperbound,' lower: ',lowerbound)
    # print('within bound?: ',lowerbound < AA_LEN[res_name] < upperbound)
    return lowerbound < AA_LEN[res_name] < upperbound

def is_below_bound(res_name, mean, std, n_std):
    """
    Returns True if given residue's 
    length is below the bound,
    else False.
    """
    lowerbound=mean-std*n_std
    return lowerbound > AA_LEN[res_name]

def get_aa_based_on_bound(res_name, mean, std, is_same_property, get_shorter_aa, n_std):
    """
    Returns list of residues within bound (standard deviation * n_std)
    (Searches all 20 residues)
    """
    aa_ls=[] 
    res_property = get_aa_property(res_name)
    upperbound=mean+std*n_std
    lowerbound=mean-std*n_std
    
    if is_same_property:
        for aa in PROPERTY_AA[res_property]:
            # prioritize same property aa within bound
            if  is_aa_within_range(aa, mean, std, n_std=1) and aa != res_name: 
                aa_ls.append(aa)
                
        # if suggestion list is empty, add one aa that is shorter than current aa
        if not aa_ls: 
            if get_shorter_aa: # for getting shorter aa
                sorted_aa_ls = sorted([(aa, AA_LEN[aa]) for aa in PROPERTY_AA[res_property]], key=lambda x:x[1]) # ascending by distance
            else: # for getting longer aa
                sorted_aa_ls = sorted([(aa, AA_LEN[aa]) for aa in PROPERTY_AA[res_property]], key=lambda x:x[1], reverse=True) # descending by distance
            
            curr_res_idx = sorted_aa_ls.index((res_name, AA_LEN[res_name]))
            # get aa one size shorter if exitsts
            if curr_res_idx > 0:
                suggested_aa = sorted_aa_ls[curr_res_idx-1]
                aa_ls.append(suggested_aa[0])
            
    else:
        all_20_res_ls = list(AA_LEN.keys())
        
        # loop all aa and find all aa within bound
        for aa in all_20_res_ls:
            if  is_aa_within_range(aa, mean, std, n_std=1) and aa != res_name: 
                aa_ls.append(aa)
                
        # if suggestion list is empty, add one aa that is shorter than current aa
        if not aa_ls: 
            if get_shorter_aa: # for getting shorter aa
                sorted_aa_ls = sorted([(aa, AA_LEN[aa]) for aa in PROPERTY_AA[res_property]], key=lambda x:x[1]) # ascending by distance
            else: # for getting longer aa
                sorted_aa_ls = sorted([(aa, AA_LEN[aa]) for aa in PROPERTY_AA[res_property]], key=lambda x:x[1], reverse=True) # descending by distance
            
            curr_res_idx = sorted_aa_ls.index((res_name, AA_LEN[res_name]))
            # get aa one size shorter as long as its not the shortest aa in the list
            if curr_res_idx > 0:
                suggested_aa = sorted_aa_ls[curr_res_idx-1]
                aa_ls.append(suggested_aa[0])
        
    return aa_ls

def get_polar_or_charged_aa_based_on_bound(res_name, mean, std, n_std):
    """
    Returns list of aa within bound (standard deviation * n_std)
    (Search polar/charged residues )
    """
    aa_ls=[]
    
    upperbound=mean+std*n_std
    lowerbound=mean-std*n_std
    
    # make polar/charged aa list
    p_c_aa_ls = []
    for aa_property in PROPERTY_AA.keys():
        if aa_property == 'polar' or aa_property == 'negative' or aa_property == 'positive':
            p_c_aa_ls+= PROPERTY_AA[aa_property]
    p_c_aa_ls.append(res_name)
    
    for aa in p_c_aa_ls:
        # prioritize same property aa within bound
        if is_aa_within_range(aa, mean, std, n_std=1) and aa != res_name: 
            aa_ls.append(aa)

    # if suggestion list is empty, add one aa that is shorter than current aa
    if not aa_ls: 
        sorted_aa_ls = sorted([(aa, AA_LEN[aa]) for aa in p_c_aa_ls], key=lambda x:x[1], reverse=True) # descending by distance
        curr_res_idx = sorted_aa_ls.index((res_name, AA_LEN[res_name]))
        # get aa one size shorter if exitsts
        if curr_res_idx > 0:
            suggested_aa = sorted_aa_ls[curr_res_idx-1]
            aa_ls.append(suggested_aa[0])
    return aa_ls

def get_nonpolar_aa_based_on_bound(res_name, mean, std, n_std):
    """
    Returns list of residues within bound (standard deviation * n_std)
    (Search non-polar residues)
    """
    aa_ls=[]
    
    upperbound=mean+std*n_std
    lowerbound=mean-std*n_std
    
    # make non-polar aa list
    nonp_aa_ls = PROPERTY_AA['nonpolar']+[res_name]
    
    for aa in nonp_aa_ls:
        # prioritize same property aa within bound
        if is_aa_within_range(aa, mean, std, n_std=1) and aa != res_name: 
            aa_ls.append(aa)

    # if suggestion list is empty, add one aa that is one size longer than current aa
    if not aa_ls: 
        sorted_aa_ls = sorted([(aa, AA_LEN[aa]) for aa in nonp_aa_ls], key=lambda x:x[1], reverse=True) # descending by distance
        curr_res_idx = sorted_aa_ls.index((res_name, AA_LEN[res_name]))
        # get aa one size shorter if exitsts
        if curr_res_idx > 0:
            suggested_aa = sorted_aa_ls[curr_res_idx-1]
            aa_ls.append(suggested_aa[0])
    return aa_ls

def suggest_sim_len_same_property_aa(res_name, mean, std):
    """
    Returns list of same property aa 
    within bound (3*standard deviation away from given mean)
    
    *If no aa found within bound, suggest 1 size longer/shorter than current given residue.
    (unless current residue is the longest/shortest)
    
    *if current residue is within the bound, no aa is suggested.
    
    * Doesn't take into acct properties of protonates states of residues
    """
    res_len = AA_LEN[res_name]
    is_within_bound = is_aa_within_range(res_name, mean, std, n_std=1)
    if not is_within_bound:
        if is_below_bound(res_name, mean, std, n_std=1):
            # suggest aa within same property group based on length
            same_prop_aa_ls = get_aa_based_on_bound(res_name, mean, std, is_same_property=True, get_shorter_aa=False, n_std=1)
        else:
            same_prop_aa_ls = get_aa_based_on_bound(res_name, mean, std, is_same_property=True, get_shorter_aa=True, n_std=1)
    else: # current aa within bound => In this function, need to strictly suggest aa with same property. So, if curr residue is within bound, don't suggest any aa.
        return [], is_within_bound
    return same_prop_aa_ls, is_within_bound

def suggest_sim_len_aa(res_name, mean, std, near_sub_h_bond_donor):
    """
    Returns list of aa within bound 
    (3*standard deviation away from given mean)
    
    *near_sub_h_bond_donor: True if h-bond donor atom is in the top3 closest substrate atoms, else False
    
    *If no aa found within bound, suggest 1 size longer/shorter than current given residue.
    (unless current residue is the longest/shortest)
    *if current residue is within the bound, no aa is suggested.
    * Doesn't take into acct properties of protonates states of residues
    """
    res_len = AA_LEN[res_name]
    is_within_bound = is_aa_within_range(res_name, mean, std, n_std=1)
    if not is_within_bound:
        if is_below_bound(res_name, mean, std, n_std=1):
            aa_ls = get_aa_based_on_bound(res_name, mean, std, is_same_property=False, get_shorter_aa=False, n_std=1)
        else:
            aa_ls = get_aa_based_on_bound(res_name, mean, std, is_same_property=False, get_shorter_aa=True, n_std=1)
    else: # current aa within bound
        # if near h-bond donor substrate atom* suggest polar/charged aa within bound
        if near_sub_h_bond_donor:
            aa_ls = get_polar_or_charged_aa_based_on_bound(res_name, mean, std, n_std=1)
        else:
            aa_ls = get_nonpolar_aa_based_on_bound(res_name, mean, std, n_std=1)
    return aa_ls, is_within_bound

def loop_sub_near_aa_to_suggest_aa(traj, PARAM, res_idx_to_aa_sub_atom_data_dict):
    """
    *finds residues for suggestion based on distance and general properties of 
    residue/substrate_atom given the filtered MD frame data.
    
    Returns 4 lists:
        1. list of residue index of residues found near substrate 
        2. list of residue name of residues found near substrate
        3. list of list: [atom idx, atom name, CA-substrate_atom mean distance, standard deviation]
        4. list of dictionary: {key=suggested_residue_name : value=residue_length}
    """
    # length of these 4 lists = number of aa found near substreate
    aa_near_sub_resi_ls = [] # column name
    aa_near_sub_resn_ls = [] 
    aa_near_sub_top3_atoms_ls = [] # element: list of tuple(atom_idx, atom_name, mean, std) 
    aa_near_sub_aa_suggestion_and_length_ls = [] # element: list of tuple(suggested resn, res length)
    
    sub_atom_idx_to_name_dict = get_sub_atom_idx_to_name_dict(traj, PARAM)
    for res_idx, data_ls in res_idx_to_aa_sub_atom_data_dict.items():
        res_name = data_ls[0][0]
        p = get_aa_property(res_name)

        aa_near_sub_resi_ls.append(res_idx+1) # make resi 1 indexing
        aa_near_sub_resn_ls.append(res_name)
        aa_near_sub_top3_atoms_ls.append([(data[3]+1, sub_atom_idx_to_name_dict[data[3]], data[1], data[2]) for data in data_ls]) # make atom index 1 indexing

        curr_res_near_sub_atom_names_ls = [sub_atom_idx_to_name_dict[data[3]] for data in data_ls]
        
        # where there is h-bond donor atom in substrate
        if 'O' in curr_res_near_sub_atom_names_ls or 'N' in curr_res_near_sub_atom_names_ls:
            # use shortest distance from list of nearby substrate atoms*
            data = data_ls[0]
            mean = data[1]
            std = data[2]
            if p in ['polar','positive','negative']: # when curr residue is polar/charged 
                aa_ls, is_within_bound = suggest_sim_len_same_property_aa(res_name, mean, std)
            else: # when curr residue is non-polar
                aa_ls, is_within_bound = suggest_sim_len_aa(res_name, mean, std, near_sub_h_bond_donor=True) # => across property, just look into distance

        else: # when near substrate atoms are all carbon
            data = data_ls[0]
            mean = data[1]
            std = data[2]
            if p in ['nonpolar']: # when curr residue is non-polar
                aa_ls, is_within_bound = suggest_sim_len_same_property_aa(res_name, mean, std)
            else: # when curr residue is polar/charged 
                aa_ls, is_within_bound = suggest_sim_len_aa(res_name, mean, std, near_sub_h_bond_donor=False) # => across property, just look into distance
        aa_near_sub_aa_suggestion_and_length_ls.append([(aa,AA_LEN[aa]) for aa in aa_ls])
        
    return aa_near_sub_resi_ls, aa_near_sub_resn_ls, aa_near_sub_top3_atoms_ls, aa_near_sub_aa_suggestion_and_length_ls

def get_final_output(aa_near_sub_resi_ls, aa_near_sub_resn_ls, aa_near_sub_top3_atoms_ls, aa_near_sub_aa_suggestion_and_length_ls):
    """
    Returns final output: suggested AA stored in dict
    => {key=resi: value=[AA replacements suggestions]}
    """
    resi_to_aa_suggestions_dict={}

    for resi, resn, atom_data_ls, aa_sugg in zip(aa_near_sub_resi_ls, aa_near_sub_resn_ls, aa_near_sub_top3_atoms_ls, aa_near_sub_aa_suggestion_and_length_ls):
        
        resi_to_aa_suggestions_dict[resi]=\
        [(resn,), \
         {atom_data[0]:[atom_data[1], atom_data[2], atom_data[3]] for atom_data in atom_data_ls }, \
         {resname:length for resname,length in aa_sugg}]
    return resi_to_aa_suggestions_dict