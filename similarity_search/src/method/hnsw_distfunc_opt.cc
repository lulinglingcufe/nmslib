/**
 * Non-metric Space Library
 *
 * Main developers: Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov, Ben Frederickson, David Novak
 *
 * For the complete list of contributors and further details see:
 * https://github.com/nmslib/nmslib
 *
 * Copyright (c) 2013-2018
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */
/*
*
* A Hierarchical Navigable Small World (HNSW) approach.
*
* The main publication is (available on arxiv: http://arxiv.org/abs/1603.09320):
* "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin
* This code was contributed by Yu. A. Malkov. It also was used in tests from the paper.
*
*
*/
#include "ztimer.h"
#include "Keccak256.h"
#include "method/hnsw.h"
#include "method/hnsw_distfunc_opt_impl_inline.h"
#include "knnquery.h"
#include "ported_boost_progress.h"
#include "rangequery.h"

#include "portable_prefetch.h"
#include "space.h"

#include "sort_arr_bi.h"
#define MERGE_BUFFER_ALGO_SWITCH_THRESHOLD 100

#include <algorithm> // std::min
#include <limits>
#include <vector>

#include <iomanip>
using std::uint8_t;
using std::setfill;
using std::setw;


//#define DIST_CALC
namespace similarity {

    int nodeCount = 0;
    //std::uint8_t * actualHashArray[1000];
    //int searchCount = 0;

    template <typename dist_t>
    void
    Hnsw<dist_t>::SearchOld(KNNQuery<dist_t> *query, bool normalize)
    {
        LOG(LIB_INFO) << "This is SearchOld: ";

        float *pVectq = (float *)((char *)query->QueryObject()->data());
        TMP_RES_ARRAY(TmpRes);
        size_t qty = query->QueryObject()->datalength() >> 2;

        if (normalize) {
            NormalizeVect(pVectq, qty);
        }

        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;

        int maxlevel1 = maxlevel_;
        int curNodeNum = enterpointId_;
        dist_t curdist = (fstdistfunc_(
            pVectq, (float *)(data_level0_memory_ + enterpointId_ * memoryPerObject_ + offsetData_ + 16), qty, TmpRes));

        for (int i = maxlevel1; i > 0; i--) {
            bool changed = true;
            while (changed) {
                changed = false;
                int *data = (int *)(linkLists_[curNodeNum] + (maxM_ + 1) * (i - 1) * sizeof(int));
                int size = *data;
                for (int j = 1; j <= size; j++) {
                    PREFETCH(data_level0_memory_ + (*(data + j)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
                }
#ifdef DIST_CALC
                query->distance_computations_ += size;
#endif

                for (int j = 1; j <= size; j++) {
                    int tnum = *(data + j);

                    dist_t d = (fstdistfunc_(
                        pVectq, (float *)(data_level0_memory_ + tnum * memoryPerObject_ + offsetData_ + 16), qty, TmpRes));
                    if (d < curdist) {
                        curdist = d;
                        curNodeNum = tnum;
                        changed = true;
                    }
                }
            }
        }

        priority_queue<EvaluatedMSWNodeInt<dist_t>> candidateQueuei; // the set of elements which we can use to evaluate

        priority_queue<EvaluatedMSWNodeInt<dist_t>> closestDistQueuei; // The set of closest found elements
        // EvaluatedMSWNodeInt<dist_t> evi(curdist, curNodeNum);
        candidateQueuei.emplace(-curdist, curNodeNum);

        closestDistQueuei.emplace(curdist, curNodeNum);

        // query->CheckAndAddToResult(curdist, new Object(data_level0_memory_ + (curNodeNum)*memoryPerObject_ + offsetData_));
        query->CheckAndAddToResult(curdist, data_rearranged_[curNodeNum]);

        //打印hash


		std::uint8_t actualHash[Keccak256::HASH_LEN];
        //2022-11-22-Keccak256方法.md: 把buffer 指针 转换为uint8_t 指针，(uint8_t *)
		Keccak256::getHash(  (uint8_t *)data_rearranged_[curNodeNum]->buffer(), data_rearranged_[curNodeNum]->bufferlength(), actualHash);

        LOG(LIB_INFO) << "actualHash: " << actualHash;



        massVisited[curNodeNum] = currentV;

        while (!candidateQueuei.empty()) {
            EvaluatedMSWNodeInt<dist_t> currEv = candidateQueuei.top(); // This one was already compared to the query

            dist_t lowerBound = closestDistQueuei.top().getDistance();
            if ((-currEv.getDistance()) > lowerBound) {
                break;
            }

            candidateQueuei.pop();
            curNodeNum = currEv.element;
            int *data = (int *)(data_level0_memory_ + curNodeNum * memoryPerObject_ + offsetLevel0_);
            int size = *data;
            PREFETCH((char *)(massVisited + *(data + 1)), _MM_HINT_T0);
            PREFETCH((char *)(massVisited + *(data + 1) + 64), _MM_HINT_T0);
            PREFETCH(data_level0_memory_ + (*(data + 1)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
            PREFETCH((char *)(data + 2), _MM_HINT_T0);

            for (int j = 1; j <= size; j++) {
                int tnum = *(data + j);
                PREFETCH((char *)(massVisited + *(data + j + 1)), _MM_HINT_T0);
                PREFETCH(data_level0_memory_ + (*(data + j + 1)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
                if (!(massVisited[tnum] == currentV)) {
#ifdef DIST_CALC
                    query->distance_computations_++;
#endif
                    nodeCount++;  //这里可以看到访问的node到底有多少个
                    massVisited[tnum] = currentV;
                    char *currObj1 = (data_level0_memory_ + tnum * memoryPerObject_ + offsetData_);
                    dist_t d = (fstdistfunc_(pVectq, (float *)(currObj1 + 16), qty, TmpRes));
                    if (closestDistQueuei.top().getDistance() > d || closestDistQueuei.size() < ef_) {
                        candidateQueuei.emplace(-d, tnum);
                        PREFETCH(data_level0_memory_ + candidateQueuei.top().element * memoryPerObject_ + offsetLevel0_,
                                     _MM_HINT_T0);
                        // query->CheckAndAddToResult(d, new Object(currObj1));
                        query->CheckAndAddToResult(d, data_rearranged_[tnum]);
                        closestDistQueuei.emplace(d, tnum);

                        if (closestDistQueuei.size() > ef_) {
                            closestDistQueuei.pop();
                        }
                    }
                }
            }
        }
        visitedlistpool->releaseVisitedList(vl);
        LOG(LIB_INFO) << "nodeCount: " << nodeCount;
    }

    template <typename dist_t>
    void
    Hnsw<dist_t>::SearchV1Merge(KNNQuery<dist_t> *query, bool normalize)
    {
        // WallClockTimer wtm;
        // wtm.reset();

        //hash
        //LOG(LIB_INFO) << "This is SearchV1Merge: ";
        //std::uint8_t actualHash[Keccak256::HASH_LEN];


        float *pVectq = (float *)((char *)query->QueryObject()->data());
        TMP_RES_ARRAY(TmpRes);
        size_t qty = query->QueryObject()->datalength() >> 2;

        if (normalize) {
            NormalizeVect(pVectq, qty);
        }

        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;

        int maxlevel1 = maxlevel_;
        int curNodeNum = enterpointId_;
        nodeCount++;

        dist_t curdist = (fstdistfunc_(
            pVectq, (float *)(data_level0_memory_ + enterpointId_ * memoryPerObject_ + offsetData_ + 16), qty, TmpRes));

        for (int i = maxlevel1; i > 0; i--) {
            bool changed = true;
            while (changed) {
                changed = false;
                int *data = (int *)(linkLists_[curNodeNum] + (maxM_ + 1) * (i - 1) * sizeof(int));
                int size = *data;
                for (int j = 1; j <= size; j++) {
                    PREFETCH(data_level0_memory_ + (*(data + j)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
                }
#ifdef DIST_CALC
                query->distance_computations_ += size;
#endif

                for (int j = 1; j <= size; j++) {
                    int tnum = *(data + j);
                    nodeCount++;

                    dist_t d = (fstdistfunc_(
                        pVectq, (float *)(data_level0_memory_ + tnum * memoryPerObject_ + offsetData_ + 16), qty, TmpRes));
                    if (d < curdist) {
                        curdist = d;
                        curNodeNum = tnum;
                        changed = true;
                    }
                }
            }
        }

        SortArrBI<dist_t, int> sortedArr(max<size_t>(ef_, query->GetK()));
        sortedArr.push_unsorted_grow(curdist, curNodeNum);

        int_fast32_t currElem = 0;

        typedef typename SortArrBI<dist_t, int>::Item QueueItem;
        vector<QueueItem> &queueData = sortedArr.get_data();
        vector<QueueItem> itemBuff(1 + max(maxM_, maxM0_));

        massVisited[curNodeNum] = currentV;

        while (currElem < min(sortedArr.size(), ef_)) {
            auto &e = queueData[currElem];
            CHECK(!e.used);
            e.used = true;
            curNodeNum = e.data;
            ++currElem;

            size_t itemQty = 0;
            dist_t topKey = sortedArr.top_key();

            int *data = (int *)(data_level0_memory_ + curNodeNum * memoryPerObject_ + offsetLevel0_);
            int size = *data;
            PREFETCH((char *)(massVisited + *(data + 1)), _MM_HINT_T0);
            PREFETCH((char *)(massVisited + *(data + 1) + 64), _MM_HINT_T0);
            PREFETCH(data_level0_memory_ + (*(data + 1)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
            PREFETCH((char *)(data + 2), _MM_HINT_T0);

            for (int j = 1; j <= size; j++) {
                int tnum = *(data + j);
                PREFETCH((char *)(massVisited + *(data + j + 1)), _MM_HINT_T0);
                PREFETCH(data_level0_memory_ + (*(data + j + 1)) * memoryPerObject_ + offsetData_, _MM_HINT_T0);
                if (!(massVisited[tnum] == currentV)) {
#ifdef DIST_CALC
                    query->distance_computations_++;
#endif
                    massVisited[tnum] = currentV;
                    nodeCount++;

                    char *currObj1 = (data_level0_memory_ + tnum * memoryPerObject_ + offsetData_);
                    dist_t d = (fstdistfunc_(pVectq, (float *)(currObj1 + 16), qty, TmpRes));

                    if (d < topKey || sortedArr.size() < ef_) {
                        CHECK_MSG(itemBuff.size() > itemQty,
                                  "Perhaps a bug: buffer size is not enough " + 
                                  ConvertToString(itemQty) + " >= " + ConvertToString(itemBuff.size()));
                        itemBuff[itemQty++] = QueueItem(d, tnum);
                    }
                }
            }

            if (itemQty) {
                PREFETCH(const_cast<const char *>(reinterpret_cast<char *>(&itemBuff[0])), _MM_HINT_T0);
                std::sort(itemBuff.begin(), itemBuff.begin() + itemQty);

                size_t insIndex = 0;
                if (itemQty > MERGE_BUFFER_ALGO_SWITCH_THRESHOLD) {
                    insIndex = sortedArr.merge_with_sorted_items(&itemBuff[0], itemQty);

                    if (insIndex < currElem) {
                        currElem = insIndex;
                    }
                } else {
                    for (size_t ii = 0; ii < itemQty; ++ii) {
                        size_t insIndex = sortedArr.push_or_replace_non_empty_exp(itemBuff[ii].key, itemBuff[ii].data);
                        if (insIndex < currElem) {
                            currElem = insIndex;
                        }
                    }
                }
                // because itemQty > 1, there would be at least item in sortedArr
                PREFETCH(data_level0_memory_ + sortedArr.top_item().data * memoryPerObject_ + offsetLevel0_, _MM_HINT_T0);
            }
            // To ensure that we either reach the end of the unexplored queue or currElem points to the first unused element
            while (currElem < sortedArr.size() && queueData[currElem].used == true)
                ++currElem;
        }

        for (int_fast32_t i = 0; i < query->GetK() && i < sortedArr.size(); ++i) {
            int tnum = queueData[i].data;
            // char *currObj = (data_level0_memory_ + tnum*memoryPerObject_ + offsetData_);
            // query->CheckAndAddToResult(queueData[i].key, new Object(currObj));
            query->CheckAndAddToResult(queueData[i].key, data_rearranged_[tnum]);

            // //打印hash结果
		    // Keccak256::getHash(  (uint8_t *)data_rearranged_[tnum]->buffer(), data_rearranged_[tnum]->bufferlength(), actualHash);

            // //存储hash结果

            // actualHashArray[searchCount] = actualHash;
            // searchCount++;




            //LOG(LIB_INFO) << "actualHash: ";
            //  for(int j = 0; j < 32; j++) {
            //      //printf("%x", REV(actualHash[j]));
            //      printf("%02X", actualHash[j]);
            //     }
            //     printf("\n");           

           

            //LOG(LIB_INFO) << hex << actualHash;
            
            // LOG(LIB_INFO) << "actualHash: ";

            // for(int j = 0; j < 32; j++) {
            //     LOG(LIB_INFO) << hex << actualHash[j];
            //     }
        }
        visitedlistpool->releaseVisitedList(vl);
        LOG(LIB_INFO) << "nodeCount: " << nodeCount;
	    nodeCount = 0;
        // wtm.split();
        // const double SearchTime  = double(wtm.elapsed())/1e6;
        // LOG(LIB_INFO) << ">>>> Search time:         " << SearchTime;   
        //时间相关的计时代码。      




        // LOG(LIB_INFO) << "actualHashArray: ";

        // for(int j = 0; j < searchCount; j++) {
        //     LOG(LIB_INFO) << "actualHash: ";
        //      for(int k = 0; k < 32; k++) {
        //          printf("%02X", actualHashArray[j][k]);
        //         }
        //         printf("\n"); 
        //         }
        // LOG(LIB_INFO) << "searchCount: " << searchCount;        


  


    }

    template class Hnsw<float>;
    template class Hnsw<int>;
}
