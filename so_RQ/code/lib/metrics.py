from scipy.stats import spearmanr,pearsonr,kendalltau
import math
import numpy as np
import warnings
from decimal import Decimal

def evaluate(test_data, predicted, option='votes'):
    ndcg_at_1 = []
    ndcg_at_5 = []
    ndcg= []
    kendall_cor = []
    pearson_cor = []
    #warnings.filterwarnings('error')
    
    for i in range(len(test_data)):
        assert test_data[i][0] == predicted[i][0]
        grank_score = test_data[i][2]
        #print grank_score
        grank_score = deduplicate(grank_score)
        prank_score = predicted[i][1]
        r = 1
        gr = []
        pr = []
        gr_score = []
        pr_score = []
        for usg in grank_score:
            #print usg
            for usp in prank_score:
                if usp[0] == usg[0]:
                    gr.append(float(usg[1]))
                    pr.append(float(usp[1]))
                    gr_score.append([usg[0],usg[1]])
                    pr_score.append([usp[0],usp[1]])
                    break
        pr_score = sorted(pr_score, key=lambda pr_score : pr_score[1], reverse=True)
        
        # following code deals with the exceptions that may occur in calculating ndcg and rank correlation
        
        if len(gr) < 2:
            #print 'inconsistent: the #answerers of a test question is less than 2!'
            continue
        conti = False
        for item in gr_score:
            if item[1]!=0:
                conti = True
        if not conti:
            continue
        
        if not nDCG_at_k(gr_score, pr_score, 1) == -1:
            ndcg_at_1.append(nDCG_at_k(gr_score, pr_score, 1, option))
        if not nDCG_at_k(gr_score, pr_score, 5) == -1:
            ndcg_at_5.append(nDCG_at_k(gr_score, pr_score, 5, option))
        if not nDCG_at_k(gr_score, pr_score, len(grank_score)) == -1:
            ndcg.append(nDCG_at_k(gr_score, pr_score, len(grank_score), option))
            '''print '[warning:] NDCG calculation encounters a problem.'
            print gr_score
            print pr_score
            print nDCG_at_k(gr_score, pr_score, 1)
            print nDCG_at_k(gr_score, pr_score, 5)
            print nDCG_at_k(gr_score, pr_score, len(grank_score))'''

        #try: 
        if isinstance(kendalltau(gr,pr),int):
            pearson_cor.append(pearsonr(gr,pr)[0])
            kendall_cor.append(kendalltau(gr,pr))
        else:
            pearson_cor.append(pearsonr(gr,pr)[0])
            kendall_cor.append(kendalltau(gr,pr)[0])
        '''except Warning: 
            print gr
            print pr'''
        
        '''print gr_score
        print pr_score
        print nDCG_at_k(gr_score, pr_score, 1, option)
        print nDCG_at_k(gr_score, pr_score, 5, option)
        print nDCG_at_k(gr_score, pr_score, len(grank_score), option)
        print gr
        print pr
        print pearsonr(gr,pr)
        print kendalltau(gr,pr)'''
        
    print '++++++++++++ evaluation +++++++++++++'
    #print pearson_cor
    pearson_cor = [value for value in pearson_cor if not math.isnan(value)]
    kendall_cor = [value for value in kendall_cor if not math.isnan(value)]
    print 'total questions evaluated - pearson: '+str(len(pearson_cor))+ '; kendall: '+str(len(kendall_cor))+'; ndcg: '+str(len(ndcg))
    print 'pearson: '+str(np.mean(pearson_cor))
    print 'kendalltau: '+str(np.mean(kendall_cor))
    print 'ndcg@1: '+str(np.mean(ndcg_at_1))
    print 'ndcg@5: '+str(np.mean(ndcg_at_5))
    print 'ndcg: '+str(np.mean(ndcg))
    print '+++++++++++++++++++++++++++++++++++++'
    return np.mean(ndcg_at_1)

def evaluate_train(test_data, predicted, option='votes'):
    ndcg_at_1 = []
    
    for i in range(len(test_data)):
        assert test_data[i][0] == predicted[i][0]
        grank_score = test_data[i][2]
        #print grank_score
        grank_score = deduplicate(grank_score)
        prank_score = predicted[i][1]
        r = 1
        gr = []
        pr = []
        gr_score = []
        pr_score = []
        for usg in grank_score:
            #print usg
            for usp in prank_score:
                if usp[0] == usg[0]:
                    gr.append(float(usg[1]))
                    pr.append(float(usp[1]))
                    gr_score.append([usg[0],usg[1]])
                    pr_score.append([usp[0],usp[1]])
                    break
        pr_score = sorted(pr_score, key=lambda pr_score : pr_score[1], reverse=True)
        
        # following code deals with the exceptions that may occur in calculating ndcg and rank correlation
        
        if len(gr) < 2:
            #print 'inconsistent: the #answerers of a test question is less than 2!'
            continue
        conti = False
        for item in gr_score:
            if item[1]!=0:
                conti = True
        if not conti:
            continue
        
        if not nDCG_at_k(gr_score, pr_score, 1) == -1:
            ndcg_at_1.append(nDCG_at_k(gr_score, pr_score, 1, option))

    return np.mean(ndcg_at_1)
    
def nDCG_at_k(gt_list, rec_list, k, option='votes'):
    rel = dict([])
    i = 0
    for us in gt_list:
        if option=='rank':
            rel[us[0]] = len(gt_list)-i
        else:
            rel[us[0]] = us[1]
        i += 1
    dcg = 0
    idcg = 0
    for i in range(len(rec_list)):
        rec = rec_list[i]
        gt = gt_list[i]
        if i==0:
            dcg += 2**rel[rec[0]]-1
            idcg += 2**rel[gt[0]]-1
        else:
            try:
                dcg += float(2**rel[rec[0]]-1)/math.log(i+2,2)
                idcg += float(2**rel[gt[0]]-1)/math.log(i+2,2)
            except:
                dcg = long(dcg)
                idcg= long(idcg)
                dcg += long(Decimal(2**rel[rec[0]]-1)/Decimal(math.log(i+2,2)))
                idcg += long(Decimal(2**rel[gt[0]]-1)/Decimal(math.log(i+2,2)))
        i += 1
        if i+1>k:
            break
    if idcg<=0:
        return -1
    else:
        try:
            return float(dcg)/idcg
        except:
            return long(Decimal(dcg)/Decimal(idcg))

def deduplicate(grank_score):
    new_grank_score = []
    u_set = set([us[0] for us in grank_score])
    for u in u_set:
        for us in grank_score:
            if u==us[0]:
                new_grank_score.append(us)
                break
    new_grank_score = sorted(new_grank_score, key=lambda new_grank_score : new_grank_score[1], reverse=True)
    return new_grank_score

def MRR():
    return 0


if __name__ == '__main__':
    grank_score = [[0,1],[1,2],[0,3],[2,2]]
    print deduplicate(grank_score)
    gt_list = [[1,3],[3,3],[2,2],[6,2],[5,1],[4,0]]
    rec_list = [[1,3],[2,2],[3,3],[4,0],[5,1],[6,2]]
    print nDCG_at_k(gt_list, rec_list, 6)
    '''gt_list = [[1,0]]
    rec_list = [[1,0]]
    print nDCG_at_k(gt_list, rec_list, 6)'''
    
    print '========'
    test_data = [[2, ['c#'], [[3, 1], [1, 2]]]]
    predicted = [[2, [[1, 10], [3, 5.496770903376116]]]]
    evaluate(test_data, predicted)