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
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <stdio.h>
#include <unistd.h>


#include "portable_prefetch.h"
#if defined(_WIN32) || defined(WIN32)
#include <intrin.h>
#endif

#include "portable_simd.h"
#include "space.h"
#include "rangequery.h"
#include "knnquery.h"
#include "searchoracle.h"
#include "method/vptree.h"
#include "method/vptree_utils.h"
#include "methodfactory.h"

#define MIN_PIVOT_SELECT_DATA_QTY 10
#define MAX_PIVOT_SELECT_ATTEMPTS 5

namespace similarity {

using std::string;
using std::stringstream;
using std::endl;
using std::cout;
using std::cerr;

const IdType PIVOT_ID_NULL_NODE = -2;
const IdType PIVOT_ID_NULL_PIVOT = -1;
const uint32_t VERSION_NUMBER = 2;

int VPtreeVisitTimes_test = 0;
int VPtreeVisitTimes_leaf = 0;
int leaf_node_number = 0;
unsigned MAX_LEVEL = 0;
//char float_array[10];
char dest[528];
std::uint8_t  internal_node_hash_value_[(Keccak256::HASH_LEN)*3];
//std::uint8_t  internal_node_hash_value_2_[(Keccak256::HASH_LEN)*3];

template <typename dist_t, typename SearchOracle>
VPTree<dist_t, SearchOracle>::VPTree(
                       bool  PrintProgress,
                       Space<dist_t>& space,
                       const ObjectVector& data,
                       bool use_random_center) :
                              Index<dist_t>(data),
                              space_(space),
                              PrintProgress_(PrintProgress),
                              use_random_center_(use_random_center),
                              max_pivot_select_attempts_(MAX_PIVOT_SELECT_ATTEMPTS),
                              oracle_(space, data, PrintProgress),
                              QueryTimeParams_(oracle_.GetQueryTimeParamNames()) { 
                                QueryTimeParams_.push_back("maxLeavesToVisit");
                              }

template <typename dist_t, typename SearchOracle>
void VPTree<dist_t, SearchOracle>::CreateIndex(const AnyParams& IndexParams) {
  AnyParamManager pmgr(IndexParams);

  pmgr.GetParamOptional("bucketSize", BucketSize_, 50);
  pmgr.GetParamOptional("chunkBucket", ChunkBucket_, true);
  pmgr.GetParamOptional("selectPivotAttempts", max_pivot_select_attempts_, MAX_PIVOT_SELECT_ATTEMPTS);

  CHECK_MSG(max_pivot_select_attempts_ >= 1, "selectPivotAttempts should be >=1");

  LOG(LIB_INFO) << "bucketSize          = " << BucketSize_;
  LOG(LIB_INFO) << "chunkBucket         = " << ChunkBucket_;
  LOG(LIB_INFO) << "selectPivotAttempts = " << max_pivot_select_attempts_;

  // Call this function *ONLY AFTER* the bucket size is obtained!
  oracle_.SetIndexTimeParams(pmgr);
  oracle_.LogParams();

  pmgr.CheckUnused();

  this->ResetQueryTimeParams(); // reset query-time parameters

  unique_ptr<ProgressDisplay>   progress_bar(PrintProgress_ ? 
                                              new ProgressDisplay(this->data_.size(), cerr):
                                              NULL);
  LOG(LIB_INFO) << "This is  root test ";
  root_.reset(new VPNode(//0,
                     progress_bar.get(), 
                     oracle_, 
                     space_, this->data_,
                     max_pivot_select_attempts_,
                     BucketSize_, ChunkBucket_,
                     NULL,
                     0,
                     use_random_center_ /* use random center */));


  //把构造hash的递归函数放在这里
  // for (unsigned i=MAX_LEVEL-1; i>=0; i--){
  // root_->RecursiveToConstructHash(i);
  // }
  //或者这里让他先睡一会儿，再来构建hash
  //sleep(5);

  
  //root_->RecursiveToConstructHash(MAX_LEVEL-1);


  //GenericConstructHash(root_);
  if (progress_bar) { // make it 100%
    (*progress_bar) += (progress_bar->expected_count() - progress_bar->count());
  }

  // float sum_value = 0;
  // sum_value = root_->GetHashValue();
  

  std::uint8_t * sum_value = root_->GetHashValue();

  //root_->RecursiveToConstructHash();  //通过递归构建hash。
  
            LOG(LIB_INFO) << "root Hash: ";
             for(int j = 0; j < 32; j++) {
                 printf("%02X", sum_value[j]);
                }
                printf("\n");   


  // for ( int i = 0; i < 2; i++ )
  //  {
  //      cout << "*(p + " << i << ") : ";
  //      cout << *(sum_value + i) << endl;
  //  } 

  //LOG(LIB_INFO) << "sum_value is          = " << sum_value;
  //相当于这里直接获取到merkle root 



}




template <typename dist_t,typename SearchOracle>
VPTree<dist_t, SearchOracle>::~VPTree() {
}

template <typename dist_t,typename SearchOracle>
const std::string VPTree<dist_t, SearchOracle>::StrDesc() const {
  return "vptree: " + SearchOracle::GetName();
}

template <typename dist_t, typename SearchOracle>
void VPTree<dist_t, SearchOracle>::Search(RangeQuery<dist_t>* query, IdType) const {
  int mx = MaxLeavesToVisit_;
  root_->GenericSearch(query, mx);
  LOG(LIB_INFO) << "RangeQuery VPtreeVisitTimes_test          = " << VPtreeVisitTimes_test;
  VPtreeVisitTimes_test = 0;  
  VPtreeVisitTimes_leaf = 0;
}

template <typename dist_t, typename SearchOracle>
void VPTree<dist_t, SearchOracle>::Search(KNNQuery<dist_t>* query, IdType) const {
  int mx = MaxLeavesToVisit_;

  LOG(LIB_INFO) << "KNNQuery leaf_node_number          = " << leaf_node_number;
  root_->GenericSearch(query, mx);
  LOG(LIB_INFO) << "KNNQuery VPtreeVisitTimes_test          = " << VPtreeVisitTimes_test;
  LOG(LIB_INFO) << "KNNQuery VPtreeVisitTimes_leaf          = " << VPtreeVisitTimes_leaf;
  VPtreeVisitTimes_test = 0;  
  VPtreeVisitTimes_leaf = 0;

  root_->RecursiveToConstructHash();//构建完整的验证merkle tree

  //打印完整的验证merkle tree
  for (unsigned i=0; i<=MAX_LEVEL; i++){
  printf("level  %d  :",i);
  root_->RecursiveToPrintHashLevel(i);
  printf("\n");
  }

  //打印发送给用户的验证merkle tree
  for (unsigned i=0; i<=MAX_LEVEL; i++){
  printf("level  %d  :",i);
  root_->RecursivePrintHashTree(i);
  printf("\n");
  }
  //root_->RecursivePrintHashTree(); 

  
            // LOG(LIB_INFO) << "root Hash: ";
            //  for(int j = 0; j < 32; j++) {
            //      printf("%02X", sum_value[j]);
            //     }
            //     printf("\n");   




}

template <typename dist_t, typename SearchOracle>
void VPTree<dist_t, SearchOracle>::SaveIndex(const std::string& location) {

  std::ofstream output(location, std::ios::binary);
  CHECK_MSG(output, "Cannot open file '" + location + "' for writing");
  output.exceptions(ios::badbit | ios::failbit);

  // Save version number
  const uint32_t version = VERSION_NUMBER;
  writeBinaryPOD(output, version);

  // Save dataset size, so that we can compare with dataset size upon loading
  writeBinaryPOD(output, this->data_.size());

  // Save tree parameters
  writeBinaryPOD(output, max_pivot_select_attempts_);
  writeBinaryPOD(output, BucketSize_);
  writeBinaryPOD(output, ChunkBucket_);
  writeBinaryPOD(output, use_random_center_);

  // Save node data
  if (root_) {
    SaveNodeData(output, root_.get());
  }
  output.close();
}

template <typename dist_t, typename SearchOracle>
void VPTree<dist_t, SearchOracle>::LoadIndex(const std::string& location) {

  std::ifstream input(location, std::ios::binary);
  CHECK_MSG(input, "Cannot open file '" + location + "' for reading");
  input.exceptions(ios::badbit | ios::failbit);

  uint32_t version;
  readBinaryPOD(input, version);
  if (version != VERSION_NUMBER) {
    PREPARE_RUNTIME_ERR(err) << "File version number (" << version << ") differs from "
                             << "expected version (" << VERSION_NUMBER << ")";
    THROW_RUNTIME_ERR(err);
  }

  size_t datasize;
  readBinaryPOD(input, datasize);
  if (datasize != this->data_.size()) {
    PREPARE_RUNTIME_ERR(err) << "Saved dataset size (" << datasize
                             << ") differs from actual size (" << this->data_.size() << ")";
    THROW_RUNTIME_ERR(err);
  }

  readBinaryPOD(input, max_pivot_select_attempts_);
  readBinaryPOD(input, BucketSize_);
  readBinaryPOD(input, ChunkBucket_);
  readBinaryPOD(input, use_random_center_);

  vector<IdType> IdMapper;

  CreateObjIdToPosMapper(this->data_, IdMapper);

  root_ = std::unique_ptr<VPNode>(LoadNodeData(input, ChunkBucket_, IdMapper));
}

template <typename dist_t, typename SearchOracle>
void
VPTree<dist_t, SearchOracle>::SaveNodeData(
  std::ofstream& output,
  const typename VPTree<dist_t, SearchOracle>::VPNode* node) const {

  // Nodes are written to output in pre-order. 
  // If a node is null, we write a sentinel value instead (PIVOT_ID_NULL_NODE).
  // Regular nodes are
  // serialized in the following order:
  //
  // * pivot ID (or PIVOT_ID_NULL_NODE or PIVOT_ID_NULL_PIVOT)
  // * median distance
  // * number of bucket elements: if the bucket is empty or null, we write zero
  // * IDs of bucket data elements
  // * left child tree
  // * right child tree

  IdType pivotId = PIVOT_ID_NULL_NODE;
  if (node != nullptr) {
    pivotId = node->pivot_ ? node->pivot_->id() : PIVOT_ID_NULL_PIVOT;
  }
  writeBinaryPOD(output, pivotId);

  if (node == nullptr) {
    return;
  }

  CHECK(node != nullptr);
  writeBinaryPOD(output, node->mediandist_);

  size_t bucket_size = node->bucket_ ? node->bucket_->size() : 0; 
  writeBinaryPOD(output, bucket_size);

  if (node->bucket_) {
    for (const auto& element : *(node->bucket_)) {
      writeBinaryPOD(output, element->id());
    }
  } 

  SaveNodeData(output, node->left_child_);
  SaveNodeData(output, node->right_child_);
}

template <typename dist_t, typename SearchOracle>
typename VPTree<dist_t, SearchOracle>::VPNode*
VPTree<dist_t, SearchOracle>::LoadNodeData(std::ifstream& input, bool ChunkBucket, const vector<IdType>& IdMapper) const {
  IdType pivotId = 0;
  // read one element, the pivot
  readBinaryPOD(input, pivotId);

  if (pivotId == PIVOT_ID_NULL_NODE) {
    // empty node
    return nullptr;
  }

  VPNode* node = new VPNode(oracle_);
  if (pivotId >= 0) {
    pivotId = ConvertId(pivotId, IdMapper);
    CHECK_MSG(pivotId >= 0, "Incorrect element ID: " + ConvertToString(pivotId));
    CHECK_MSG(pivotId < this->data_.size(), "Incorrect element ID: " + ConvertToString(pivotId));
    node->pivot_ = this->data_[pivotId];
  } else {
    CHECK(pivotId == PIVOT_ID_NULL_PIVOT);
  }

  float mediandist;
  readBinaryPOD(input, mediandist);
  node->mediandist_ = mediandist;

  size_t bucketsize;
  readBinaryPOD(input, bucketsize);

  if (bucketsize) {
    // read bucket content
    ObjectVector bucket(bucketsize);
    IdType dataId = 0;
    for (size_t i = 0; i < bucketsize; i++) {
      readBinaryPOD(input, dataId);
      CHECK(dataId >= 0);
      dataId = ConvertId(dataId, IdMapper);
      CHECK_MSG(dataId >= 0, "Incorrect element ID: " + ConvertToString(dataId));
      CHECK_MSG(dataId < this->data_.size(), "Incorrect element ID: " + ConvertToString(dataId));
      bucket[i] = this->data_[dataId];
    }
    node->CreateBucket(ChunkBucket, bucket, nullptr);
  }

  node->left_child_ = LoadNodeData(input, ChunkBucket, IdMapper);
  node->right_child_ = LoadNodeData(input, ChunkBucket, IdMapper);

  return node;
}

template <typename dist_t, typename SearchOracle>
void VPTree<dist_t, SearchOracle>::VPNode::CreateBucket(bool ChunkBucket, 
                                                        const ObjectVector& data, 
                                                        ProgressDisplay* progress_bar) {
    if (ChunkBucket) {
      CreateCacheOptimizedBucket(data, CacheOptimizedBucket_, bucket_);
    } else {
      bucket_ = new ObjectVector(data);
    }
    if (progress_bar) (*progress_bar) += data.size();
}

template <typename dist_t, typename SearchOracle>
VPTree<dist_t, SearchOracle>::VPNode::VPNode(const SearchOracle& oracle)
    : oracle_(oracle),
      pivot_(NULL),
      mediandist_(0),
      left_child_(NULL),
      right_child_(NULL),
      bucket_(NULL),
      father_node_(NULL),
      if_set_node_hash(false),
      //node_hash_value_(0),
      //node_id_(0),
      CacheOptimizedBucket_(NULL) {}

template <typename dist_t, typename SearchOracle>
VPTree<dist_t, SearchOracle>::VPNode::VPNode(
                               //unsigned level,
                               ProgressDisplay* progress_bar,
                               const SearchOracle& oracle,
                               const Space<dist_t>& space, const ObjectVector& data,
                               size_t max_pivot_select_attempts,
                               size_t BucketSize, bool ChunkBucket,
                               VPNode* father_node_point_,
                               //int node_id_,
                               unsigned level,
                               bool use_random_center)
    : oracle_(oracle),
      pivot_(NULL), mediandist_(0),
      left_child_(NULL), right_child_(NULL), father_node_(father_node_point_),
      level(level),
      if_set_node_hash(false),
      //node_hash_value_(0),
      bucket_(NULL), CacheOptimizedBucket_(NULL)
{
  CHECK(!data.empty());

  if (!data.empty() && data.size() <= BucketSize) {
    //如果是叶子节点
    //LOG(LIB_INFO) << "We have a leaf node:  "<< this;// <<"  with size :   " <<data.size() <<" with space:    "<<TotalSpaceUsed(data); 
    CreateBucket(ChunkBucket, data, progress_bar);
    Keccak256::getHash(  (uint8_t *)CacheOptimizedBucket_, TotalSpaceUsed(data), actualHash);
    leaf_node_number++;
  //打印actualHash值
  // for(int j = 0; j < 32; j++) {
  // printf("%02X", actualHash[j]);
  // }
  // printf("\n"); 

    return;
  }

  if (level > MAX_LEVEL){
    MAX_LEVEL = level;
    //LOG(LIB_INFO) << " VPNode MAX_LEVEL:   "<< MAX_LEVEL ;
  }


  if (data.size() >= 2) {
    unsigned bestDP = 0;
    float    largestSIGMA = 0;
    vector<DistObjectPairVector<dist_t>> dpARR(max_pivot_select_attempts);

    // To compute StdDev we need at least 2 points not counting the pivot
    size_t maxAtt = data.size() >= max(3, MIN_PIVOT_SELECT_DATA_QTY) ? max_pivot_select_attempts : 1;

    vector<double> dists(data.size());
    for (size_t att = 0; att < maxAtt; ++att) {
      dpARR[att].reserve(data.size());
      const size_t  currPivotIndex = SelectVantagePoint(data, use_random_center);
      const Object* pCurrPivot = data[currPivotIndex];
      for (size_t i = 0; i < data.size(); ++i) {
        if (i == currPivotIndex) {
          continue;
        }
        // Distance can be asymmetric, the pivot is always on the left side!
        dist_t d = space.IndexTimeDistance(pCurrPivot, data[i]);
        dists[i] = d;
        dpARR[att].emplace_back(d, data[i]);
      }

      double sigma = StdDev(&dists[0], dists.size());
      if (att == 0 || sigma > largestSIGMA) {
        //LOG(LIB_INFO) << " ### " << largestSIGMA << " -> "  << sigma << " att=" << att << " data.size()=" << data.size();
        largestSIGMA = sigma;
        bestDP = att;
        pivot_ = pCurrPivot;
        std::sort(dpARR[att].begin(), dpARR[att].end(), DistObjectPairAscComparator<dist_t>());
      }
    }


    DistObjectPairVector<dist_t>& dp = dpARR[bestDP];
    DistObjectPair<dist_t>  medianDistObj = GetMedian(dp);
    mediandist_ = medianDistObj.first; 

    ObjectVector left;
    ObjectVector right;

    for (auto it = dp.begin(); it != dp.end(); ++it) {
      const Object* v = it->second;

      /* 
       * Note that here we compare a pair (distance, pointer)
       * If distances are equal, pointers are compared.
       * Thus, we would get a balanced split, even if the median
       * occurs many times in the array dp[].
       */
      if (*it < medianDistObj) {
        left.push_back(v);
      } else {
        right.push_back(v);
      }
    }

    /*
     * Sometimes, e.g.., for integer-valued distances,
     * mediandist_ will be non-discriminative. In this case
     * it is more efficient to put everything into a single bucket.
     */
    size_t LeastSize = dp.size() / BalanceConst;
    //LOG(LIB_INFO) << "Construct actualHash "<< this;
  //在这里构建merkle的输入：制高点、中间值:这里好奇怪？
	char float_array[10];
	sprintf(float_array, "%f", mediandist_);
  //char dest[528];
  strcpy(dest, pivot_->buffer());
  memcpy(dest+518, float_array, 10);
  //打印hash结果
  Keccak256::getHash(  (uint8_t *)dest, 528, actualHash);
  //LOG(LIB_INFO) << "pivot_->bufferlength()         = " << pivot_->bufferlength();//都是528  
  //打印actualHash值
  // for(int j = 0; j < 32; j++) {
  // printf("%02X", actualHash[j]);
  // }
  // printf("\n"); 






    if (left.size() < LeastSize || right.size() < LeastSize) {
        CreateBucket(ChunkBucket, data, progress_bar);
        return;
    }

    if (!left.empty()) {
      //LOG(LIB_INFO) << "Construct left_child_ "<< this;
      left_child_ = new VPNode(progress_bar, oracle_, space, left, max_pivot_select_attempts, BucketSize, ChunkBucket, this, level + 1, use_random_center);


    }

    if (!right.empty()) {
      //LOG(LIB_INFO) << "Construct right_child_ "<< this;
      right_child_ = new VPNode(progress_bar, oracle_, space, right, max_pivot_select_attempts, BucketSize, ChunkBucket, this, level + 1, use_random_center);



    }




    
  } else {
    LOG(LIB_INFO) << "We have a leaf node:  "<< this;
    CHECK_MSG(data.size() == 1, "Bug: expect the subset to contain exactly one element!");
    pivot_ = data[0];
      //在这里构建merkle的输入：制高点、中间值:这里好奇怪？
	  char float_array[10];
	sprintf(float_array, "%f", mediandist_);
  //char dest[528];
  strcpy(dest, pivot_->buffer());
  memcpy(dest+518, float_array, 10);
  //打印hash结果
  Keccak256::getHash(  (uint8_t *)dest, 528, actualHash);
  //LOG(LIB_INFO) << "pivot_->bufferlength()         = " << pivot_->bufferlength();//都是528  
  //打印actualHash值
  for(int j = 0; j < 32; j++) {
  printf("%02X", actualHash[j]);
  }
  printf("\n"); 

  }
   
 
}

template <typename dist_t, typename SearchOracle>
VPTree<dist_t, SearchOracle>::VPNode::~VPNode() {
  delete left_child_;
  delete right_child_;
  delete father_node_;
  ClearBucket(CacheOptimizedBucket_, bucket_);
}

template <typename dist_t, typename SearchOracle>
template <typename QueryType>
void VPTree<dist_t, SearchOracle>::VPNode::GenericSearch(QueryType* query,
                                                         int& MaxLeavesToVisit) {



  if (MaxLeavesToVisit <= 0) return; // early termination

   VPtreeVisitTimes_test++; //每查询一次就自增一次。
  //LOG(LIB_INFO) << "GenericSearch VPtreeVisitTimes_test          = " << VPtreeVisitTimes_test;
  if_set_node_hash = true;
  if (bucket_) { //如果是叶子节点
    VPtreeVisitTimes_leaf++;

    --MaxLeavesToVisit;

    if (CacheOptimizedBucket_) {
      PREFETCH(CacheOptimizedBucket_, _MM_HINT_T0);
    }

    for (unsigned i = 0; i < bucket_->size(); ++i) {
      const Object* Obj = (*bucket_)[i];
      dist_t distQC = query->DistanceObjLeft(Obj);
      query->CheckAndAddToResult(distQC, Obj);
    }
    return;
  }

  // Distance can be asymmetric, the pivot is always the left argument (see the function that creates the node)!
  dist_t distQC = query->DistanceObjLeft(pivot_);
  query->CheckAndAddToResult(distQC, pivot_);

  if (distQC < mediandist_) {      // the query is inside
    // then first check inside
    if (left_child_ != NULL && oracle_.Classify(distQC, query->Radius(), mediandist_) != kVisitRight)
       left_child_->GenericSearch(query, MaxLeavesToVisit);

    /* 
     * After potentially visiting the left child, we need to reclassify the node,
     * because the query radius might have decreased.
     */



    // after that outside
    if (right_child_ != NULL && oracle_.Classify(distQC, query->Radius(), mediandist_) != kVisitLeft)
       right_child_->GenericSearch(query, MaxLeavesToVisit);
  } else {                         // the query is outside
    // then first check outside
    if (right_child_ != NULL && oracle_.Classify(distQC, query->Radius(), mediandist_) != kVisitLeft)
       right_child_->GenericSearch(query, MaxLeavesToVisit);

    /* 
     * After potentially visiting the left child, we need to reclassify the node,
     * because the query radius might have decreased.
     */

    // after that inside
    if (left_child_ != NULL && oracle_.Classify(distQC, query->Radius(), mediandist_) != kVisitRight)
      left_child_->GenericSearch(query, MaxLeavesToVisit);
  }
}

template <typename dist_t, typename SearchOracle>
//GenericConstructHash函数的实现  
std::uint8_t * VPTree<dist_t, SearchOracle>::VPNode::GetHashValue() {

  //LOG(LIB_INFO) << "Construct GetHashValue "<< this;
  //LOG(LIB_INFO) << "Construct GetHashValue father_node_ ："<< this->father_node_;

  if(left_child_ == NULL){

    if(right_child_ != NULL){
    LOG(LIB_INFO) << " VPNode only have  right_child_";
  }
  }
  if(right_child_ == NULL){

  //  LOG(LIB_INFO) << " This is leaf node Hash for:  "<< this; 
  //            for(int j = 0; j < 32; j++) {
  //                printf("%02X", actualHash[j]);
  //               }
  //               printf("\n");  

    memcpy(node_hash_value_, actualHash, Keccak256::HASH_LEN);
 





    if(left_child_ != NULL){
    LOG(LIB_INFO) << " VPNode only have  left_child_";
  }
  }

  if (left_child_ != NULL  && right_child_ != NULL){

    std::uint8_t  internal_node_hash_value_[(Keccak256::HASH_LEN)*2];
    memcpy(internal_node_hash_value_, left_child_->GetHashValue(), Keccak256::HASH_LEN);
    memcpy(internal_node_hash_value_+ Keccak256::HASH_LEN, right_child_->GetHashValue(), Keccak256::HASH_LEN);
    Keccak256::getHash(internal_node_hash_value_, (Keccak256::HASH_LEN)*2, node_hash_value_);

            //  LOG(LIB_INFO) << "This is internal node Hash: ";
            //  for(int j = 0; j < 32; j++) {
            //      printf("%02X", node_hash_value_[j]);
            //     }
            //     printf("\n");   

    //在这里增加拼接hash的操作。

  }
  return node_hash_value_; 

  }

template <typename dist_t, typename SearchOracle>
//template <typename QueryType>
void VPTree<dist_t, SearchOracle>::VPNode::RecursiveToPrintHashLevel(unsigned i) {
  //用这个函数把hash tree打印出来？

   if(left_child_ == NULL){
    if (if_set_node_hash && level == i){
             for(int j = 0; j < 1; j++) {
                 printf("%02X", node_hash_value_[j]);
                }
                printf("----");
    }
  }
  else if(left_child_ != NULL  && right_child_ != NULL){
    if(if_set_node_hash && level == i){
             for(int j = 0; j < 1; j++) {
                 printf("%02X", node_hash_value_[j]);
                }
                printf("----");
    }
    if(level < i){
    left_child_->RecursiveToPrintHashLevel(i);
    right_child_->RecursiveToPrintHashLevel(i);
    }

  }
}


template <typename dist_t, typename SearchOracle>
//template <typename QueryType>
void VPTree<dist_t, SearchOracle>::VPNode::RecursiveToConstructHash() {
  //用这个函数把hash tree打印出来？
if(if_set_node_hash && father_node_ !=NULL){

if (father_node_->left_child_->if_set_node_hash == false){
  father_node_->left_child_->if_set_node_hash = true;
}
if (father_node_->right_child_->if_set_node_hash == false){
  father_node_->right_child_->if_set_node_hash = true;
}

}

 if(left_child_ != NULL  && right_child_ != NULL){
    left_child_->RecursiveToConstructHash();
    right_child_->RecursiveToConstructHash();

  }
}







template <typename dist_t, typename SearchOracle>
//template <typename QueryType>
int VPTree<dist_t, SearchOracle>::VPNode::RecursivePrintHashTree(unsigned i) {
   if(left_child_ == NULL){
    if (if_set_node_hash){
      LOG(LIB_INFO) << "This is hash in level :  "<< level;
             for(int j = 0; j < 32; j++) {
                 printf("%02X", node_hash_value_[j]);
                }
                printf("\n");        
      return 1;
    }
    return 0;
  }
  else if(left_child_ != NULL  && right_child_ != NULL){
    int left = left_child_->RecursivePrintHashTree();
    int right = right_child_->RecursivePrintHashTree();
  
   if (left==0 && right==0 ){
    if (if_set_node_hash){
      LOG(LIB_INFO) << "This is internal hash in level :  "<< level;
             for(int j = 0; j < 32; j++) {
                 printf("%02X", node_hash_value_[j]);
                }
                printf("\n");        
      return 1;      
    }
    return 0;
   }
   return 1;   
  }
}






template class VPTree<float, PolynomialPruner<float> >;
template class VPTree<int, PolynomialPruner<int> >;

// namespace similarity
}
