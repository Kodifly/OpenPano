// File: stitcher.cc
// Date: Wed Nov 22 15:45:13 2017 +0800
// Author: Yuxin Wu


#include "stitcher.hh"

#include <limits>
#include <string>
#include <cmath>
#include <queue>

#include "feature/matcher.hh"
#include "lib/imgproc.hh"
#include "lib/timer.hh"
#include "blender.hh"
#include "match_info.hh"
#include "transform_estimate.hh"
#include "camera_estimator.hh"
#include "camera.hh"
#include "warp.hh"
using namespace std;
using namespace pano;
using namespace config;

namespace pano {

// use in development
const static bool DEBUG_OUT = false;
const static char* MATCHINFO_DUMP = "log/matchinfo.txt";

Mat32f Stitcher::build() {
  calc_feature();
  // TODO choose a better starting point by MST use centrality

  pairwise_matches.resize(imgs.size());
  for (auto& k : pairwise_matches) k.resize(imgs.size());
  if (ORDERED_INPUT)
    linear_pairwise_match();
  else
    pairwise_match();
  free_feature();
  //load_matchinfo(MATCHINFO_DUMP);
  if (DEBUG_OUT) {
    draw_matchinfo();
    dump_matchinfo(MATCHINFO_DUMP);
  }
  assign_center();

  if (ESTIMATE_CAMERA)
    estimate_camera();
  else
    build_linear_simple();		// naive mode
  pairwise_matches.clear();
  // TODO automatically determine projection method even in naive mode
  if (ESTIMATE_CAMERA)
    bundle.proj_method = ConnectedImages::ProjectionMethod::spherical;
  else
    bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
  print_debug("Using projection method: %d\n", bundle.proj_method);
  bundle.update_proj_range();

  return bundle.blend();
}

bool Stitcher::match_image(
    const PairWiseMatcher& pwmatcher, int i, int j) {
  auto match = pwmatcher.match(i, j);
  TransformEstimation transf(
      match, keypoints[i], keypoints[j],
      imgs[i].shape(), imgs[j].shape());	// from j to i. H(p_j) ~= p_i
  MatchInfo info;
  bool succ = transf.get_transform(&info);
  if (!succ) {
    if (-(int)info.confidence >= 8)	// reject for geometry reason
      print_debug("Reject bad match with %d inlier from %d to %d\n",
          -(int)info.confidence, i, j);
    return false;
  }
  auto inv = info.homo.inverse();	// TransformEstimation ensures invertible
  inv.mult(1.0 / inv[8]);	// TODO more stable?
  print_debug(
      "Connection between image %d and %d, ninliers=%lu/%d=%lf, conf=%f\n",
      i, j, info.match.size(), match.size(),
      info.match.size() * 1.0 / match.size(),
      info.confidence);

  // fill in pairwise matches
  pairwise_matches[i][j] = info;
  info.homo = inv;
  info.reverse();
  pairwise_matches[j][i] = move(info);
  return true;
}

void Stitcher::pairwise_match() {
  GuardedTimer tm("pairwise_match()");
  size_t n = imgs.size();
  vector<pair<int, int>> tasks;
  REP(i, n) REPL(j, i + 1, n) tasks.emplace_back(i, j);

  PairWiseMatcher pwmatcher(feats);

  int total_nr_match = 0;

#pragma omp parallel for schedule(dynamic)
  REP(k, (int)tasks.size()) {
    int i = tasks[k].first, j = tasks[k].second;
    bool succ = match_image(pwmatcher, i, j);
    if (succ)
      total_nr_match += pairwise_matches[i][j].match.size();
  }
  print_debug("Total number of matched keypoint pairs: %d\n", total_nr_match);
}

void Stitcher::linear_pairwise_match() {
  GuardedTimer tm("linear_pairwise_match()");
  int n = imgs.size();
  PairWiseMatcher pwmatcher(feats);
#pragma omp parallel for schedule(dynamic)
  REP(i, n) {
    int next = (i + 1) % n;
    if (!match_image(pwmatcher, i, next)) {
      if (i == n - 1)	// head and tail don't have to match
        continue;
      else
        error_exit(ssprintf("Image %d and %d don't match\n", i, next));
    }
    continue; // TODO FIXME a->b, b->a
    do {
      next = (next + 1) % n;
      if (next == i)
        break;
    } while (match_image(pwmatcher, i, next));
  }
}

void Stitcher::assign_center() {
  // bundle.identity_idx = imgs.size() >> 1;
  bundle.identity_idx = 5;
}

void Stitcher::estimate_camera() {
  vector<Shape2D> shapes;
  for (auto& m: imgs) shapes.emplace_back(m.shape());
  int identity_idx = bundle.identity_idx;
  // auto cameras = CameraEstimator{pairwise_matches, shapes, identity_idx}.estimate();

  // // ------------------- OUTPUTTING TRANSFORMATIONS -------------------

  // cout << "--- Transformations to Identity Camera (" << identity_idx << ") ---" << endl;

  // for (size_t i = 0; i < cameras.size(); ++i) {
  //   Homography R_identity_to_i = cameras[i].R;
  //   Homography R_i_to_identity = R_identity_to_i.inverse();

  //   cout << "\nTransformation from Camera " << i << " to Identity:" << endl;
    
  //   // Simply use the overloaded << operator
  //   cout << R_i_to_identity << endl;
  // }
  // // --------------------------------------------------------------------
  
  // // produced homo operates on [-w/2,w/2] coordinate
  // REP(i, imgs.size()) {
  //   cout << "Camera " << i << setprecision(10) << " " << cameras[i].R << ", " << cameras[i].K() << endl;
  //   bundle.component[i].homo_inv = cameras[i].K() * cameras[i].R;
  //   bundle.component[i].homo = cameras[i].Rinv() * cameras[i].K().inverse();
  // }
  
  double arr_0[9] = { -0.9985888946, -0.0124487552, 0.0516260411,
                      -0.0127924356,  0.9998981244, -0.0063320210,
                      -0.0515419559, -0.0069835087, -0.9986464126 };
  double arr_1[9] = {  0.4621649974, -0.0028250856, -0.8867894531,
                      0.0010322108,  0.9999959619, -0.0026477790,
                      0.8867933523,  0.0003083571,  0.4621660472 };
  double arr_2[9] = {  0.5403707286,  0.0117520165,  0.8413449743,
                      -0.0191471486,  0.9998152853, -0.0016679094,
                      -0.8412091669, -0.0152080678,  0.5404959318 };
  double arr_3[9] = { -0.5070286379, -0.0074376857, -0.8618971175,
                      -0.0253613345,  0.9996585432,  0.0062928356,
                      0.8615560127,  0.0250495090, -0.5070441391 };
  double arr_4[9] = { -0.4572679830, -0.0050358700,  0.8893146978,
                      -0.0123106030,  0.9999239988, -0.0006676572,
                      -0.8892437465, -0.0112532985, -0.4572952247 };
  double arr_5[9] = {  1.0000000000, -0.0000000000,  0.0000000000,
                      0.0000000000,  1.0000000000, -0.0000000000,
                      -0.0000000000,  0.0000000000,  1.0000000000 };

  Homography R_identity_to_0(arr_0);
  Homography R_identity_to_1(arr_1);
  Homography R_identity_to_2(arr_2);
  Homography R_identity_to_3(arr_3);
  Homography R_identity_to_4(arr_4);
  Homography R_identity_to_5(arr_5);

  double karr_0[9] = { 1423.6510436091, 0.0000000000, -30.8335314730,
                      0.0000000000,   1423.6510436091, 2.6317322927,
                      0.0000000000,   0.0000000000,   1.0000000000 };
  double karr_1[9] = { 1416.1851759822, 0.0000000000, 92.5113853641,
                      0.0000000000,   1416.1851759822, -9.2591240533,
                      0.0000000000,   0.0000000000,   1.0000000000 };
  double karr_2[9] = { 1442.8160626614, 0.0000000000, -8.1432683616,
                      0.0000000000,   1442.8160626614, 3.6693237945,
                      0.0000000000,   0.0000000000,   1.0000000000 };
  double karr_3[9] = { 1368.9686742328, 0.0000000000, -24.3564658952,
                      0.0000000000,   1368.9686742328, 16.4258463643,
                      0.0000000000,   0.0000000000,   1.0000000000 };
  double karr_4[9] = { 1432.3598194127, 0.0000000000, 18.1474713050,
                      0.0000000000,   1432.3598194127, 1.1116189196,
                      0.0000000000,   0.0000000000,   1.0000000000 };
  double karr_5[9] = { 1433.6267770317, 0.0000000000, -58.0127637401,
                      0.0000000000,   1433.6267770317, -5.4157515520,
                      0.0000000000,   0.0000000000,   1.0000000000 };

  Homography K_0(karr_0);
  Homography K_1(karr_1);
  Homography K_2(karr_2);
  Homography K_3(karr_3);
  Homography K_4(karr_4);
  Homography K_5(karr_5);

  bundle.component[0].homo_inv = K_0 * R_identity_to_0;
  bundle.component[0].homo = R_identity_to_0.transpose() * K_0.inverse();
  bundle.component[1].homo_inv = K_1 * R_identity_to_1;
  bundle.component[1].homo = R_identity_to_1.transpose() * K_1.inverse();
  bundle.component[2].homo_inv = K_2 * R_identity_to_2;
  bundle.component[2].homo = R_identity_to_2.transpose() * K_2.inverse();
  bundle.component[3].homo_inv = K_3 * R_identity_to_3;
  bundle.component[3].homo = R_identity_to_3.transpose() * K_3.inverse();
  bundle.component[4].homo_inv = K_4 * R_identity_to_4;
  bundle.component[4].homo = R_identity_to_4.transpose() * K_4.inverse();
  bundle.component[5].homo_inv = K_5 * R_identity_to_5;
  bundle.component[5].homo = R_identity_to_5.transpose() * K_5.inverse();
}

void Stitcher::build_linear_simple() {
  // TODO bfs over pairwise to build bundle
  // assume pano pairwise
  int n = imgs.size(), mid = bundle.identity_idx;
  bundle.component[mid].homo = Homography::I();

  auto& comp = bundle.component;

  // accumulate the transformations
  if (mid + 1 < n) {
    comp[mid+1].homo = pairwise_matches[mid][mid+1].homo;
    REPL(k, mid + 2, n)
      comp[k].homo = comp[k - 1].homo * pairwise_matches[k-1][k].homo;
  }
  if (mid - 1 >= 0) {
    comp[mid-1].homo = pairwise_matches[mid][mid-1].homo;
    REPD(k, mid - 2, 0)
      comp[k].homo = comp[k + 1].homo * pairwise_matches[k+1][k].homo;
  }
  // comp[k]: from k to identity. [-w/2,w/2]

  // when estimate_camera is not used, homo is KRRK(2d-2d), not KR(2d-3d)
  // need to somehow normalize(guess) focal length to make non-flat projection work
  double f = -1;
  if (not TRANS)    // the estimation method only works under fixed-center projection
    f = Camera::estimate_focal(pairwise_matches);
  if (f <= 0) {
    print_debug("Cannot estimate focal. Will use a naive one.\n");
    f = 0.5 * (imgs[mid].width() + imgs[mid].height());
  }
  REP(i, n) {
    auto M = Homography{{
        1.0/f, 0,     0,
        0,     1.0/f, 0,
        0,     0,     1
    }};
    comp[i].homo = M * comp[i].homo;
  }
  bundle.calc_inverse_homo();
}

}	// namepsace pano

