from datasets import load_ft
from utils import hypergraph_utils as hgut


def load_feature_construct_H(data_dir,
                             m_prob=1,
                             K_neigs=[10],
                             is_probH=True,
                             split_diff_scale=False,
                             use_mvcnn_feature=False,
                             use_gvcnn_feature=True,
                             use_mvcnn_feature_for_structure=False,
                             use_gvcnn_feature_for_structure=True):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """
    # init feature
    if use_mvcnn_feature or use_mvcnn_feature_for_structure:
        mvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='MVCNN')
    if use_gvcnn_feature or use_gvcnn_feature_for_structure:
        gvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='GVCNN')
    if 'mvcnn_ft' not in dir() and 'gvcnn_ft' not in dir():
        raise Exception('None feature initialized')

    # construct feature matrix
    fts = None
    if use_mvcnn_feature:
        fts = hgut.feature_concat(fts, mvcnn_ft)
    if use_gvcnn_feature:
        fts = hgut.feature_concat(fts, gvcnn_ft)
    if fts is None:
        raise Exception(f'None feature used for model!')

    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    mvcnn_dist, gvcnn_dist = None, None
    if use_mvcnn_feature_for_structure:
        mvcnn_dist = hgut.cal_distance_map(mvcnn_ft)
    if use_gvcnn_feature_for_structure:
        gvcnn_dist = hgut.cal_distance_map(mvcnn_ft)
    if mvcnn_dist is None and gvcnn_dist is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')

    return fts, lbls, idx_train, idx_test, mvcnn_dist, gvcnn_dist


def randomedge_sample_H(mvcnn_dist, gvcnn_dist, K_neigs=[10], is_probH=True,m_prob=1,percent=1,
                        split_diff_scale=False, use_mvcnn_feature_for_structure=False,
                        use_gvcnn_feature_for_structure=True):
    # construct hypergraph incidence matrix
    # print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    if use_mvcnn_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(mvcnn_dist, K_neigs=K_neigs,
                                    split_diff_scale=split_diff_scale,
                                    is_probH=is_probH, m_prob=m_prob,percent=percent)  # 用k-nn 算法生成邻接矩阵 H（由距离矩阵生成 [2012,2012]
        H = hgut.hyperedge_concat(H, tmp)
    if use_gvcnn_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(gvcnn_dist, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob,percent=percent)  # [2012,2012]
        H = hgut.hyperedge_concat(H, tmp)  # [2012,4024]
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')

    return H
