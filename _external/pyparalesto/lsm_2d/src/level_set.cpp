/*
  Copyright (c) 2015-2016 Lester Hedges <lester.hedges+lsm@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

/*! \file LevelSet.cpp
    \brief A class for the level set function.
 */

#include "level_set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <map>
#include <ostream>
#include <utility>
#include <cassert>

#include "debug.h"
#include "fast_marching_method.h"

namespace lsm_2d {

LevelSet::LevelSet (Mesh& mesh_, double moveLimit_, unsigned int bandWidth_, bool isFixed_)
    : moveLimit (moveLimit_),
      mesh (mesh_),
      bandWidth (bandWidth_),
      isFixed (isFixed_),
      isTarget (false) {
  int size = 0.2 * mesh.nNodes;

  errno = EINVAL;
  lsm_check (bandWidth > 2, "Width of the narrow band must be greater than 2.");
  lsm_check (((moveLimit > 0) && (moveLimit <= 1)), "Move limit must be between 0 and 1.");

  // Resize data structures.
  signedDistance.resize (mesh.nNodes);
  velocity.resize (mesh.nNodes);
  gradient.resize (mesh.nNodes);
  narrowBand.resize (mesh.nNodes);

  // Make sure that memory is sufficient (for small test systems).
  size = std::max (25, size);
  mines.resize (size);

  // Generate a Swiss cheese structure.
  initialise ();

  // Initialise the narrow band.
  initialiseNarrowBand ();

  return;

error:
  exit (EXIT_FAILURE);
}

LevelSet::LevelSet (Mesh& mesh_, const std::vector<Hole>& holes, double moveLimit_,
                    unsigned int bandWidth_, bool isFixed_)
    : moveLimit (moveLimit_),
      mesh (mesh_),
      bandWidth (bandWidth_),
      isFixed (isFixed_),
      isTarget (false) {
  int size = 0.2 * mesh.nNodes;

  errno = EINVAL;
  lsm_check (bandWidth > 2, "Width of the narrow band must be greater than 2.");
  lsm_check (((moveLimit > 0) && (moveLimit <= 1)), "Move limit must be between 0 and 1.");

  // Resize data structures.
  signedDistance.resize (mesh.nNodes);
  velocity.resize (mesh.nNodes);
  gradient.resize (mesh.nNodes);
  narrowBand.resize (mesh.nNodes);

  // Make sure that memory is sufficient (for small test systems).
  size = std::max (25, size);
  mines.resize (size);

  // Initialise level set function from hole array.
  initialise (holes);

  // Initialise the narrow band.
  initialiseNarrowBand ();

  return;

error:
  exit (EXIT_FAILURE);
}

LevelSet::LevelSet (Mesh& mesh_, const std::vector<Coord>& points, double moveLimit_,
                    unsigned int bandWidth_, bool isFixed_)
    : moveLimit (moveLimit_),
      mesh (mesh_),
      bandWidth (bandWidth_),
      isFixed (isFixed_),
      isTarget (false) {
  int size = 0.2 * mesh.nNodes;

  errno = EINVAL;
  lsm_check (bandWidth > 2, "Width of the narrow band must be greater than 2.");
  lsm_check (((moveLimit > 0) && (moveLimit <= 1)), "Move limit must be between 0 and 1.");

  // Resize data structures.
  signedDistance.resize (mesh.nNodes);
  velocity.resize (mesh.nNodes);
  gradient.resize (mesh.nNodes);
  narrowBand.resize (mesh.nNodes);

  // Make sure that memory is sufficient (for small test systems).
  size = std::max (25, size);
  mines.resize (size);

  // Initialise level set function from point array.
  initialise (points);

  // Initialise the narrow band.
  initialiseNarrowBand ();

  return;

error:
  exit (EXIT_FAILURE);
}

LevelSet::LevelSet (Mesh& mesh_, const std::vector<Hole>& initialHoles,
                    const std::vector<Hole>& targetHoles, double moveLimit_,
                    unsigned int bandWidth_, bool isFixed_)
    : moveLimit (moveLimit_),
      mesh (mesh_),
      bandWidth (bandWidth_),
      isFixed (isFixed_),
      isTarget (true) {
  int size = 0.2 * mesh.nNodes;

  errno = EINVAL;
  lsm_check (bandWidth > 2, "Width of the narrow band must be greater than 2.");
  lsm_check (((moveLimit > 0) && (moveLimit <= 1)), "Move limit must be between 0 and 1.");

  // Resize data structures.
  signedDistance.resize (mesh.nNodes);
  velocity.resize (mesh.nNodes);
  gradient.resize (mesh.nNodes);
  target.resize (mesh.nNodes);
  narrowBand.resize (mesh.nNodes);

  // Make sure that memory is sufficient (for small test systems).
  size = std::max (25, size);
  mines.resize (size);

  // Initialise level set function the target holes vector.
  initialise (targetHoles);
  reinitialise ();

  // Copy into target signed distance function.
  target = signedDistance;

  // Initialise level set function the initial holes vector.
  initialise (initialHoles);

  // Initialise the narrow band.
  initialiseNarrowBand ();

  return;

error:
  exit (EXIT_FAILURE);
}

LevelSet::LevelSet (Mesh& mesh_, const std::vector<Hole>& holes, const std::vector<Coord>& points,
                    double moveLimit_, unsigned int bandWidth_, bool isFixed_)
    : moveLimit (moveLimit_),
      mesh (mesh_),
      bandWidth (bandWidth_),
      isFixed (isFixed_),
      isTarget (true) {
  int size = 0.2 * mesh.nNodes;

  errno = EINVAL;
  lsm_check (bandWidth > 2, "Width of the narrow band must be greater than 2.");
  lsm_check (((moveLimit > 0) && (moveLimit <= 1)), "Move limit must be between 0 and 1.");

  // Resize data structures.
  signedDistance.resize (mesh.nNodes);
  velocity.resize (mesh.nNodes);
  gradient.resize (mesh.nNodes);
  target.resize (mesh.nNodes);
  narrowBand.resize (mesh.nNodes);

  // Make sure that memory is sufficient (for small test systems).
  size = std::max (25, size);
  mines.resize (size);

  // Initialise level set function from point array.
  initialise (points);
  reinitialise ();

  // Copy into target signed distance function.
  target = signedDistance;

  // Initialise level set function from hole array.
  initialise (holes);

  // Initialise the narrow band.
  initialiseNarrowBand ();

  return;

error:
  exit (EXIT_FAILURE);
}

LevelSet::LevelSet (Mesh& mesh_, const std::vector<Coord>& initialPoints,
                    const std::vector<Coord>& targetPoints, double moveLimit_,
                    unsigned int bandWidth_, bool isFixed_)
    : moveLimit (moveLimit_),
      mesh (mesh_),
      bandWidth (bandWidth_),
      isFixed (isFixed_),
      isTarget (true) {
  int size = 0.2 * mesh.nNodes;

  errno = EINVAL;
  lsm_check (bandWidth > 2, "Width of the narrow band must be greater than 2.");
  lsm_check (((moveLimit > 0) && (moveLimit <= 1)), "Move limit must be between 0 and 1.");

  // Resize data structures.
  signedDistance.resize (mesh.nNodes);
  velocity.resize (mesh.nNodes);
  gradient.resize (mesh.nNodes);
  target.resize (mesh.nNodes);
  narrowBand.resize (mesh.nNodes);

  // Make sure that memory is sufficient (for small test systems).
  size = std::max (25, size);
  mines.resize (size);

  // Initialise level set function from target point array.
  initialise (targetPoints);
  reinitialise ();

  // Copy into target signed distance function.
  target = signedDistance;

  // Initialise level set function from initialisation points array.
  initialise (initialPoints);

  // Initialise the narrow band.
  initialiseNarrowBand ();

  return;

error:
  exit (EXIT_FAILURE);
}

bool LevelSet::update (double timeStep) {
  // Loop over all nodes in the narrow band.
  for (unsigned int i = 0; i < nNarrowBand; i++) {
    unsigned int node = narrowBand[i];
    signedDistance[node] -= timeStep * gradient[node] * velocity[node];

    // If node is on domain boundary.
    if (mesh.nodes[node].isDomain) {
      // Enforce boundary condition.
      if (signedDistance[node] > 0) signedDistance[node] = 0;
    }

    // Reset the number of boundary points.
    mesh.nodes[node].nBoundaryPoints = 0;
  }

  // Check mine nodes.
  for (unsigned int i = 0; i < nMines; i++) {
    // Boundary is within one grid spacing of the mine.
    if (std::abs (signedDistance[mines[i]]) < 1.0) {
      // Reinitialise the signed distance function.
      reinitialise ();

      return true;
    }
  }

  return false;
}

bool LevelSet::update_no_WENO (double timeStep) {
  // Loop over all nodes in the narrow band.
  for (unsigned int i = 0; i < nNarrowBand; i++) {
    unsigned int node = narrowBand[i];
    signedDistance[node] -= timeStep * velocity[node];

    // If node is on domain boundary.
    if (mesh.nodes[node].isDomain) {
      // Enforce boundary condition (don't).
      if (signedDistance[node] > 0) {
        // signedDistance[node] = 0;
      }
    }

    // Reset the number of boundary points.
    mesh.nodes[node].nBoundaryPoints = 0;
  }

  // Check mine nodes.
  for (unsigned int i = 0; i < nMines; i++) {
    // Boundary is within one grid spacing of the mine.
    if (std::abs (signedDistance[mines[i]]) < 1.0) {
      // Reinitialise the signed distance function.
      reinitialise ();

      return true;
    }
  }

  return false;
}

void LevelSet::killNodes (const std::vector<Coord>& points) {
  // Kill nodes in a rectangular region.
  if (points.size () == 2) {
    // Loop over all nodes.
    for (unsigned int i = 0; i < mesh.nNodes; i++) {
      // Point is inside the rectangle.
      if (mesh.nodes[i].coord.x > points[0].x && mesh.nodes[i].coord.y > points[0].y
          && mesh.nodes[i].coord.x < points[1].x && mesh.nodes[i].coord.y < points[1].y) {
        signedDistance[i] = -1e-6;
        mesh.nodes[i].isFixed = true;
      }
    }
  }

  // Mask off a piece-wise linear shape.
  else {
    // Loop over all nodes.
    for (unsigned int i = 0; i < mesh.nNodes; i++) {
      // Point is inside the polygon.
      if (isInsidePolygon (mesh.nodes[i].coord, points)) {
        signedDistance[i] = -1e-6;
        mesh.nodes[i].isFixed = true;
      }
    }
  }
}

void LevelSet::killNodes (const Coord& point, const double radius) {
  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Point is inside the circle.
    if (pow(mesh.nodes[i].coord.x - point.x, 2) + pow(mesh.nodes[i].coord.y - point.y, 2) < pow(radius, 2)) {
      signedDistance[i] = -1e-6;
      mesh.nodes[i].isFixed = true;
    }
  }
}

void LevelSet::fixNodes (const std::vector<Coord>& points) {
  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Point is inside the rectangle.
    if (mesh.nodes[i].coord.x > points[0].x && mesh.nodes[i].coord.y > points[0].y
        && mesh.nodes[i].coord.x < points[1].x && mesh.nodes[i].coord.y < points[1].y) {
      mesh.nodes[i].isFixed = true;
    }
  }
}

void LevelSet::fixNodes (const Coord& point, const double radius) {
  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Point is inside the circle.
    if (pow(mesh.nodes[i].coord.x - point.x, 2) + pow(mesh.nodes[i].coord.y - point.y, 2) < pow(radius, 2)) {
      mesh.nodes[i].isFixed = true;
    }
  }
}

void LevelSet::createLevelSetBoundary (const std::vector<Coord>& points) {
  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Point is inside the rectangle.
    if (mesh.nodes[i].coord.x > points[0].x && mesh.nodes[i].coord.y > points[0].y
        && mesh.nodes[i].coord.x < points[1].x && mesh.nodes[i].coord.y < points[1].y) {
      signedDistance[i] = 0;
    }
  }
}

void LevelSet::computeVelocities (const std::vector<BoundaryPoint>& boundaryPoints) {
  // Initialise velocity (map boundary points to boundary nodes).
  initialiseVelocities (boundaryPoints);

  // Initialise fast marching method object.
  FastMarchingMethod fmm (mesh, false);

  // Reinitialise the signed distance function.
  fmm.march (signedDistance, velocity);
}

double LevelSet::computeVelocities (std::vector<BoundaryPoint>& boundaryPoints, double& timeStep,
                                    const double temperature, MersenneTwister& rng) {
  // Square root of two times temperature, sqrt(2T).
  double sqrt2T = std::sqrt (2.0 * temperature);

  // Time step scale factor.
  double scale = 1.0;

  // Check that noise won't lead to severe CFL violation.
  if ((sqrt2T * std::sqrt (timeStep)) > (0.5 * moveLimit)) {
    // Calculate time step scale factor.
    scale = (8.0 * timeStep * temperature) / (moveLimit * moveLimit);

    // Scale the time step.
    timeStep /= scale;
  }

  // Calculate noise prefactor.
  double noise = sqrt2T / std::sqrt (timeStep);

  // Add random noise to velocity of each boundary point.
  for (unsigned int i = 0; i < boundaryPoints.size (); i++)
    boundaryPoints[i].velocity += noise * rng.normal (0, 1);

  // Perform velocity extension.
  computeVelocities (boundaryPoints);

  return scale;
}

void LevelSet::computeGradients () {
  // Compute gradient of the signed distance function using upwind finite difference.
  // This function assumes that velocities have already been calculated.

  // Reset gradients.
  std::fill (gradient.begin (), gradient.end (), 0.0);

  // std::cout << "\n\ngradient.size() = " << gradient.size() << std::endl ;
  // std::cout << "\n\nnNarrowBand = " << nNarrowBand << std::endl ;

  // Loop over all nodes in the narrow band region.
  for (unsigned int i = 0; i < nNarrowBand; i++) {
    // Compute the nodal gradient.
    unsigned int node = narrowBand[i];
    // std::cout << "\nx" << std::endl ;
    gradient[node] = computeGradient (node);
    // std::cout << "\ny" << std::endl ;
  }
}

void LevelSet::reinitialise () {
  // Initialise fast marching method object.
  FastMarchingMethod fmm (mesh, false);

  // Reinitialise the signed distance function.
  fmm.march (signedDistance);

  // Reinitialise the narrow band.
  initialiseNarrowBand ();
}

void LevelSet::initialise () {
  // Generate a swiss cheese arrangement of holes.
  // The holes have a default radius of 5.

  // Number of holes in x and y directions.
  unsigned int nx = std::round ((double)mesh.width / 30);
  unsigned int ny = std::round ((double)mesh.height / 30);

  // Number of holes.
  unsigned int n1 = (nx * ny);              // outer grid
  unsigned int n2 = ((nx - 1) * (ny - 1));  // inner grid
  unsigned int nHoles = n1 + n2;

  // Initialise a vector of holes.
  std::vector<Hole> holes (nHoles);

  // Hole separations.
  double dx, dy;

  // Check that mesh is large enough.
  lsm_check (((nx > 2) && (ny > 2)), "Mesh is too small for Swiss cheese initialisation.");

  // Initialise hole separations.
  dx = ((double)mesh.width / (2 * nx));
  dy = ((double)mesh.height / (2 * ny));

  // Calculate hole coordinates. (outer grid)
  for (unsigned int i = 0; i < n1; i++) {
    // Work out x and y indices for hole.
    unsigned x = i % nx;
    unsigned int y = int(i / nx);

    // Set hole coordinates and radius.
    holes[i].coord.x = dx + (2 * x * dx);
    holes[i].coord.y = dy + (2 * y * dy);
    holes[i].r = 5;
  }

  // Calculate hole coordinates. (inner grid)
  for (unsigned int i = 0; i < n2; i++) {
    // Work out x and y indices for hole.
    unsigned x = i % (nx - 1);
    unsigned int y = int(i / (nx - 1));

    // Set hole coordinates and radius.
    holes[i + n1].coord.x = 2 * (dx + (x * dx));
    holes[i + n1].coord.y = 2 * (dy + (y * dy));
    holes[i + n1].r = 5;
  }

  // Now pass the holes to the initialise method.
  initialise (holes);

  return;

error:
  exit (EXIT_FAILURE);
}

void LevelSet::initialise (const std::vector<Hole>& holes) {
  // First initialise the signed distance based on domain boundary.
  closestDomainBoundary ();

  /* Now test signed distance against the surface of each hole.
     Update signed distance function when distance to hole surface
     is less than the current value. Since this is only done once, we
     use the simplest implementation possible.
   */

  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Loop over all holes.
    for (unsigned int j = 0; j < holes.size (); j++) {
      if (!holes[j].is_square_hole) {
        // Work out x and y distance of the node from the hole centre.
        double dx = holes[j].coord.x - mesh.nodes[i].coord.x;
        double dy = holes[j].coord.y - mesh.nodes[i].coord.y;

        // Work out distance (Pythag).
        double dist = std::sqrt (dx * dx + dy * dy);

        // Signed distance from the hole surface.
        dist -= holes[j].r;

        // If distance is less than current value, then update.
        if (dist < signedDistance[i]) signedDistance[i] = dist;
      } else {
        // Loop over all holes.
        for (unsigned int j = 0; j < holes.size (); j++) {
          // Work out minumum distance from the corners of nodes.
          double dx = holes[j].coord.x - mesh.nodes[i].coord.x;
          double dy = holes[j].coord.y - mesh.nodes[i].coord.y;

          // Work out distance (Pythag).
          double dist = 1e10;
          dist = std::min (mesh.nodes[i].coord.x - holes[j].coord1.x, dist);
          dist = std::min (mesh.nodes[i].coord.y - holes[j].coord1.y, dist);
          dist = std::min (-mesh.nodes[i].coord.x + holes[j].coord2.x, dist);
          dist = std::min (-mesh.nodes[i].coord.y + holes[j].coord2.y, dist);

          dist = -dist;

          // If distance is less than current value, then update.
          if (dist < signedDistance[i]) signedDistance[i] = dist;
        }
      }
    }
  }
}

void LevelSet::initialise (const std::vector<Coord>& points) {
  // First initialise the signed distance based on domain boundary.
  closestDomainBoundary ();

  /* The points define a piece-wise linear interface, i.e. they are
     assumed to be ordered (clockwise) with each pair of points
     forming a segment of the boundary. The first and last point
     in the vector should be identical, i.e. we have a closed loop
     (for an n-gon there would be n+1 points).

     The signed distance at each node is tested against each segment.

     N.B. We currently only support single closed interface.
   */

  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Loop over all interface segments.
    for (unsigned int j = 0; j < points.size () - 1; j++) {
      // Compute the minimum distance to the line segment j --> j+1.
      double dist = pointToLineDistance (points[j], points[j + 1], mesh.nodes[i].coord);

      // If distance is less than current value, then update.
      if (dist < signedDistance[i]) signedDistance[i] = dist;
    }

    // Invert the signed distance function if the point lies inside the polygon.
    if (isInsidePolygon (mesh.nodes[i].coord, points)) signedDistance[i] *= -1;
  }
}

void LevelSet::closestDomainBoundary () {
  // Initial LSF is distance from closest domain boundary.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Closest edge in x.
    unsigned int minX = std::min (mesh.nodes[i].coord.x, mesh.width - mesh.nodes[i].coord.x);

    // Closest edge in y.
    unsigned int minY = std::min (mesh.nodes[i].coord.y, mesh.height - mesh.nodes[i].coord.y);

    // Signed distance is the minimum of minX and minY;
    signedDistance[i] = double(std::min (minX, minY));
  }
}

void LevelSet::initialiseNarrowBand () {
  unsigned int mineWidth = bandWidth - 1;

  // Reset the number of nodes in the narrow band.
  nNarrowBand = 0;

  // Reset the number of mines.
  nMines = 0;

  // Loop over all nodes.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    // Flag node as inactive.
    mesh.nodes[i].isActive = false;

    /* Check that the node isn't in a masked region. If it's not, then check
       whether it lies on the domain boundary, and if it does then check that
       the boundary isn't fixed.
     */
    if (!mesh.nodes[i].isFixed && (!mesh.nodes[i].isDomain || !isFixed)) {
      // Absolute value of the signed distance function.
      double absoluteSignedDistance = std::abs (signedDistance[i]);

      // Node lies inside band.
      if (absoluteSignedDistance < bandWidth) {
        // Flag node as active.
        mesh.nodes[i].isActive = true;

        // Update narrow band array.
        narrowBand[nNarrowBand] = i;

        // Increment number of nodes.
        nNarrowBand++;

        // Node lines at edge of band.
        if (absoluteSignedDistance > mineWidth) {
          // Node is a mine.
          mesh.nodes[i].isMine = true;

          // Update mine array.
          mines[nMines] = i;

          // Increment mine count.
          nMines++;

          // TODO:
          // Check when number of mines exceeds array size!
        }
      }
    }
  }
}

void LevelSet::initialiseVelocities (const std::vector<BoundaryPoint>& boundaryPoints) {
  // Map boundary point velocities to nodes of the level set domain
  // using inverse squared distance interpolation.

  // Whether the velocity at a node has been set.
  bool isSet[mesh.nNodes];

  // Weighting factor for each node.
  double weight[mesh.nNodes];

  // Initialise arrays.
  for (unsigned int i = 0; i < mesh.nNodes; i++) {
    isSet[i] = false;
    weight[i] = 0;
    velocity[i] = 0;
  }

  // Loop over all boundary points.
  for (unsigned int i = 0; i < boundaryPoints.size (); i++) {
    // Find the closest node.
    unsigned int node = mesh.getClosestNode (boundaryPoints[i].coord);

    // Distance from the boundary point to the node.
    double dx = mesh.nodes[node].coord.x - boundaryPoints[i].coord.x;
    double dy = mesh.nodes[node].coord.y - boundaryPoints[i].coord.y;

    // Squared distance.
    double rSqd = dx * dx + dy * dy;

    // If boundary point lies exactly on the node, then set velocity
    // to that of the boundary point.
    if (rSqd < 1e-6) {
      velocity[node] = boundaryPoints[i].velocity;
      weight[node] = 1.0;
      isSet[node] = true;
    } else {
      // Update velocity estimate if not already set.
      if (!isSet[node]) {
        velocity[node] += boundaryPoints[i].velocity / rSqd;
        weight[node] += 1.0 / rSqd;
      }
    }

    // Loop over all neighbours of the node.
    for (unsigned int j = 0; j < 4; j++) {
      // Index of the neighbouring node.
      unsigned int neighbour = mesh.nodes[node].neighbours[j];

      // Make sure neighbour is in bounds.
      if (neighbour < mesh.nNodes) {
        // Distance from the boundary point to the node.
        double dx = mesh.nodes[neighbour].coord.x - boundaryPoints[i].coord.x;
        double dy = mesh.nodes[neighbour].coord.y - boundaryPoints[i].coord.y;

        // Squared distance.
        double rSqd = dx * dx + dy * dy;

        // If boundary point lies exactly on the node, then set velocity
        // to that of the boundary point.
        if (rSqd < 1e-6) {
          velocity[neighbour] = boundaryPoints[i].velocity;
          weight[neighbour] = 1.0;
          isSet[neighbour] = true;
        } else if (rSqd <= 1.0) {
          // Update velocity estimate if not already set.
          if (!isSet[neighbour]) {
            velocity[neighbour] += boundaryPoints[i].velocity / rSqd;
            weight[neighbour] += 1.0 / rSqd;
          }
        }
      }
    }
  }

  // Compute interpolated velocity.
  for (unsigned int i = 0; i < nNarrowBand; i++) {
    unsigned int node = narrowBand[i];
    if (velocity[node]) velocity[node] /= weight[node];
  }
}

double LevelSet::computeGradient (const unsigned int node) {
  // Nodal coordinates.
  unsigned int x = mesh.nodes[node].coord.x;
  unsigned int y = mesh.nodes[node].coord.y;

  // Nodal signed distance.
  double lsf = signedDistance[node];

  // Whether gradient has been computed.
  bool isGradient = false;

  // Zero the gradient.
  double grad = 0;

  // Node is on the left edge.
  if (x == 0) {
    // Node is at bottom left corner.
    if (y == 0) {
      // If signed distance at nodes to right and above is the same, then use
      // the diagonal node for computing the gradient.
      if ((std::abs (signedDistance[mesh.xyToIndex[x + 1][y]] - lsf) < 1e-6)
          && (std::abs (signedDistance[mesh.xyToIndex[x][y + 1]] - lsf) < 1e-6)) {
        // Calculate signed distance to diagonal node.
        grad = std::abs (lsf - signedDistance[mesh.xyToIndex[x + 1][y + 1]]);
        grad *= std::sqrt (2.0);
        isGradient = true;
      }
    }

    // Node is at top left corner.
    else if (y == mesh.height) {
      // If signed distance at nodes to right and below is the same, then use
      // the diagonal node for computing the gradient.
      if ((std::abs (signedDistance[mesh.xyToIndex[x + 1][y]] - lsf) < 1e-6)
          && (std::abs (signedDistance[mesh.xyToIndex[x][y - 1]] - lsf) < 1e-6)) {
        // Calculate signed distance to diagonal node.
        grad = std::abs (lsf - signedDistance[mesh.xyToIndex[x + 1][y - 1]]);
        grad *= std::sqrt (2.0);
        isGradient = true;
      }
    }
  }

  // Node is on the right edge.
  else if (x == mesh.width) {
    // Node is at bottom right corner.
    if (y == 0) {
      // If signed distance at nodes to left and above is the same, then use
      // the diagonal node for computing the gradient.
      if ((std::abs (signedDistance[mesh.xyToIndex[x - 1][y]] - lsf) < 1e-6)
          && (std::abs (signedDistance[mesh.xyToIndex[x][y + 1]] - lsf) < 1e-6)) {
        // Calculate signed distance to diagonal node.
        grad = std::abs (lsf - signedDistance[mesh.xyToIndex[x - 1][y + 1]]);
        grad *= std::sqrt (2.0);
        isGradient = true;
      }
    }

    // Node is at top right corner.
    else if (y == mesh.height) {
      // If signed distance at nodes to left and below is the same, then use
      // the diagonal node for computing the gradient.
      if ((std::abs (signedDistance[mesh.xyToIndex[x - 1][y]] - lsf) < 1e-6)
          && (std::abs (signedDistance[mesh.xyToIndex[x][y - 1]] - lsf) < 1e-6)) {
        // Calculate signed distance to diagonal node.
        grad = std::abs (lsf - signedDistance[mesh.xyToIndex[x - 1][y - 1]]);
        grad *= std::sqrt (2.0);
        isGradient = true;
      }
    }
  }

  // Gradient hasn't already been calculated.
  if (!isGradient) {
    // Stencil values for the WENO approximation.
    double v1, v2, v3, v4, v5;

    // Upwind direction.
    int sign = velocity[node] < 0 ? -1 : 1;

    // Derivatives to right.

    // Node on left-hand edge.
    if (x == 0) {
      v1 = signedDistance[mesh.xyToIndex[3][y]] - signedDistance[mesh.xyToIndex[2][y]];
      v2 = signedDistance[mesh.xyToIndex[2][y]] - signedDistance[mesh.xyToIndex[1][y]];
      v3 = signedDistance[mesh.xyToIndex[1][y]] - signedDistance[mesh.xyToIndex[0][y]];

      // Approximate derivatives outside of domain.
      v4 = v3;
      v5 = v3;
    }

    // One node to right of left-hand edge.
    else if (x == 1) {
      v1 = signedDistance[mesh.xyToIndex[4][y]] - signedDistance[mesh.xyToIndex[3][y]];
      v2 = signedDistance[mesh.xyToIndex[3][y]] - signedDistance[mesh.xyToIndex[2][y]];
      v3 = signedDistance[mesh.xyToIndex[2][y]] - signedDistance[mesh.xyToIndex[1][y]];
      v4 = signedDistance[mesh.xyToIndex[1][y]] - signedDistance[mesh.xyToIndex[0][y]];

      // Approximate derivatives outside of domain.
      v5 = v4;
    }

    // Node on right-hand edge.
    else if (x == mesh.width) {
      v5 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];

      // Approximate derivatives outside of domain.
      v3 = v4;
      v2 = v4;
      v1 = v4;
    }

    // One node to left of right-hand edge.
    else if (x == (mesh.width - 1)) {
      v5 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];
      v3 = signedDistance[mesh.xyToIndex[x + 1][y]] - signedDistance[mesh.xyToIndex[x][y]];

      // Approximate derivatives outside of domain.
      v2 = v3;
      v1 = v3;
    }

    // Two nodes to left of right-hand edge.
    else if (x == (mesh.width - 2)) {
      v5 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];
      v3 = signedDistance[mesh.xyToIndex[x + 1][y]] - signedDistance[mesh.xyToIndex[x][y]];
      v2 = signedDistance[mesh.xyToIndex[x + 2][y]] - signedDistance[mesh.xyToIndex[x + 1][y]];

      // Approximate derivatives outside of domain.
      v1 = v2;
    }

    // Node lies in bulk.
    else {
      v1 = signedDistance[mesh.xyToIndex[x + 3][y]] - signedDistance[mesh.xyToIndex[x + 2][y]];
      v2 = signedDistance[mesh.xyToIndex[x + 2][y]] - signedDistance[mesh.xyToIndex[x + 1][y]];
      v3 = signedDistance[mesh.xyToIndex[x + 1][y]] - signedDistance[mesh.xyToIndex[x][y]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];
      v5 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
    }

    double gradRight = sign * gradHJWENO (v1, v2, v3, v4, v5);

    // Derivatives to left.

    // Node on right-hand edge.
    if (x == mesh.width) {
      v1 = signedDistance[mesh.xyToIndex[x - 2][y]] - signedDistance[mesh.xyToIndex[x - 3][y]];
      v2 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
      v3 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];

      // Approximate derivatives outside of domain.
      v4 = v3;
      v5 = v3;
    }

    // One node to left of right-hand edge.
    else if (x == (mesh.width - 1)) {
      v1 = signedDistance[mesh.xyToIndex[x - 2][y]] - signedDistance[mesh.xyToIndex[x - 3][y]];
      v2 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
      v3 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];
      v4 = signedDistance[mesh.xyToIndex[x + 1][y]] - signedDistance[mesh.xyToIndex[x][y]];

      // Approximate derivatives outside of domain.
      v5 = v4;
    }

    // Node on left-hand edge.
    else if (x == 0) {
      v5 = signedDistance[mesh.xyToIndex[2][y]] - signedDistance[mesh.xyToIndex[1][y]];
      v4 = signedDistance[mesh.xyToIndex[1][y]] - signedDistance[mesh.xyToIndex[0][y]];

      // Approximate derivatives outside of domain.
      v3 = v4;
      v2 = v4;
      v1 = v4;
    }

    // One node to right of left-hand edge.
    else if (x == 1) {
      v5 = signedDistance[mesh.xyToIndex[3][y]] - signedDistance[mesh.xyToIndex[2][y]];
      v4 = signedDistance[mesh.xyToIndex[2][y]] - signedDistance[mesh.xyToIndex[1][y]];
      v3 = signedDistance[mesh.xyToIndex[1][y]] - signedDistance[mesh.xyToIndex[0][y]];

      // Approximate derivatives outside of domain.
      v2 = v3;
      v1 = v3;
    }

    // Two nodes to right of left-hand edge.
    else if (x == 2) {
      v5 = signedDistance[mesh.xyToIndex[4][y]] - signedDistance[mesh.xyToIndex[3][y]];
      v4 = signedDistance[mesh.xyToIndex[3][y]] - signedDistance[mesh.xyToIndex[2][y]];
      v3 = signedDistance[mesh.xyToIndex[2][y]] - signedDistance[mesh.xyToIndex[1][y]];
      v2 = signedDistance[mesh.xyToIndex[1][y]] - signedDistance[mesh.xyToIndex[0][y]];

      // Approximate derivatives outside of domain.
      v1 = v2;
    }

    // Node lies in bulk.
    else {
      v1 = signedDistance[mesh.xyToIndex[x - 2][y]] - signedDistance[mesh.xyToIndex[x - 3][y]];
      v2 = signedDistance[mesh.xyToIndex[x - 1][y]] - signedDistance[mesh.xyToIndex[x - 2][y]];
      v3 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x - 1][y]];
      v4 = signedDistance[mesh.xyToIndex[x + 1][y]] - signedDistance[mesh.xyToIndex[x][y]];
      v5 = signedDistance[mesh.xyToIndex[x + 2][y]] - signedDistance[mesh.xyToIndex[x + 1][y]];
    }

    double gradLeft = sign * gradHJWENO (v1, v2, v3, v4, v5);

    // Upward derivatives.

    // Node on bottom edge.
    if (y == 0) {
      v1 = signedDistance[mesh.xyToIndex[x][3]] - signedDistance[mesh.xyToIndex[x][2]];
      v2 = signedDistance[mesh.xyToIndex[x][2]] - signedDistance[mesh.xyToIndex[x][1]];
      v3 = signedDistance[mesh.xyToIndex[x][1]] - signedDistance[mesh.xyToIndex[x][0]];

      // Approximate derivatives outside of domain.
      v4 = v3;
      v5 = v3;
    }

    // One node above bottom edge.
    else if (y == 1) {
      v1 = signedDistance[mesh.xyToIndex[x][4]] - signedDistance[mesh.xyToIndex[x][3]];
      v2 = signedDistance[mesh.xyToIndex[x][3]] - signedDistance[mesh.xyToIndex[x][2]];
      v3 = signedDistance[mesh.xyToIndex[x][2]] - signedDistance[mesh.xyToIndex[x][1]];
      v4 = signedDistance[mesh.xyToIndex[x][1]] - signedDistance[mesh.xyToIndex[x][0]];

      // Approximate derivatives outside of domain.
      v5 = v4;
    }

    // Node is on top edge.
    else if (y == mesh.height) {
      v5 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];

      // Approximate derivatives outside of domain.
      v3 = v4;
      v2 = v4;
      v1 = v4;
    }

    // One node below top edge.
    else if (y == (mesh.height - 1)) {
      v5 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];
      v3 = signedDistance[mesh.xyToIndex[x][y + 1]] - signedDistance[mesh.xyToIndex[x][y]];

      // Approximate derivatives outside of domain.
      v2 = v3;
      v1 = v3;
    }

    // Two nodes below top edge.
    else if (y == (mesh.height - 2)) {
      v5 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];
      v3 = signedDistance[mesh.xyToIndex[x][y + 1]] - signedDistance[mesh.xyToIndex[x][y]];
      v2 = signedDistance[mesh.xyToIndex[x][y + 2]] - signedDistance[mesh.xyToIndex[x][y + 1]];

      // Approximate derivatives outside of domain.
      v1 = v2;
    }

    // Node lies in bulk.
    else {
      v1 = signedDistance[mesh.xyToIndex[x][y + 3]] - signedDistance[mesh.xyToIndex[x][y + 2]];
      v2 = signedDistance[mesh.xyToIndex[x][y + 2]] - signedDistance[mesh.xyToIndex[x][y + 1]];
      v3 = signedDistance[mesh.xyToIndex[x][y + 1]] - signedDistance[mesh.xyToIndex[x][y]];
      v4 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];
      v5 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
    }

    double gradUp = sign * gradHJWENO (v1, v2, v3, v4, v5);

    // Downward derivative.

    // Node on top edge.
    if (y == mesh.height) {
      v1 = signedDistance[mesh.xyToIndex[x][y - 2]] - signedDistance[mesh.xyToIndex[x][y - 3]];
      v2 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
      v3 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];

      // Approximate derivatives outside of domain.
      v4 = v3;
      v5 = v3;
    }

    // One node below top edge.
    else if (y == (mesh.height - 1)) {
      v1 = signedDistance[mesh.xyToIndex[x][y - 2]] - signedDistance[mesh.xyToIndex[x][y - 3]];
      v2 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
      v3 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];
      v4 = signedDistance[mesh.xyToIndex[x][y + 1]] - signedDistance[mesh.xyToIndex[x][y]];

      // Approximate derivatives outside of domain.
      v5 = v4;
    }

    // Node lies on bottom edge
    else if (y == 0) {
      v5 = signedDistance[mesh.xyToIndex[x][2]] - signedDistance[mesh.xyToIndex[x][1]];
      v4 = signedDistance[mesh.xyToIndex[x][1]] - signedDistance[mesh.xyToIndex[x][0]];

      // Approximate derivatives outside of domain.
      v3 = v4;
      v2 = v4;
      v1 = v4;
    }

    // One node above bottom edge.
    else if (y == 1) {
      v5 = signedDistance[mesh.xyToIndex[x][3]] - signedDistance[mesh.xyToIndex[x][2]];
      v4 = signedDistance[mesh.xyToIndex[x][2]] - signedDistance[mesh.xyToIndex[x][1]];
      v3 = signedDistance[mesh.xyToIndex[x][1]] - signedDistance[mesh.xyToIndex[x][0]];

      // Approximate derivatives outside of domain.
      v2 = v3;
      v1 = v3;
    }

    // Two nodes above bottom edge.
    else if (y == 2) {
      v5 = signedDistance[mesh.xyToIndex[x][4]] - signedDistance[mesh.xyToIndex[x][3]];
      v4 = signedDistance[mesh.xyToIndex[x][3]] - signedDistance[mesh.xyToIndex[x][2]];
      v3 = signedDistance[mesh.xyToIndex[x][2]] - signedDistance[mesh.xyToIndex[x][1]];
      v2 = signedDistance[mesh.xyToIndex[x][1]] - signedDistance[mesh.xyToIndex[x][0]];

      // Approximate derivatives outside of domain.
      v1 = v2;
    }

    // Node lies in bulk.
    else {
      v1 = signedDistance[mesh.xyToIndex[x][y - 2]] - signedDistance[mesh.xyToIndex[x][y - 3]];
      v2 = signedDistance[mesh.xyToIndex[x][y - 1]] - signedDistance[mesh.xyToIndex[x][y - 2]];
      v3 = signedDistance[mesh.xyToIndex[x][y]] - signedDistance[mesh.xyToIndex[x][y - 1]];
      v4 = signedDistance[mesh.xyToIndex[x][y + 1]] - signedDistance[mesh.xyToIndex[x][y]];
      v5 = signedDistance[mesh.xyToIndex[x][y + 2]] - signedDistance[mesh.xyToIndex[x][y + 1]];
    }

    double gradDown = sign * gradHJWENO (v1, v2, v3, v4, v5);

    // Compute gradient using upwind scheme.

    if (gradDown > 0) grad += gradDown * gradDown;
    if (gradLeft > 0) grad += gradLeft * gradLeft;
    if (gradUp < 0) grad += gradUp * gradUp;
    if (gradRight < 0) grad += gradRight * gradRight;

    grad = std::sqrt (grad);
  }

  // Return gradient.
  return grad;
}

double LevelSet::gradHJWENO (double v1, double v2, double v3, double v4, double v5) {
  // Approximate the gradient using the 5th order Hamilton-Jacobi WENO approximation.
  // Taken from pages 34-35 of "Level Set Methods and Dynamic Implicit Surfaces".
  // See: http://web.stanford.edu/class/cs237c/Lecture16.pdf

  double oneQuarter = 1.0 / 4.0;
  double thirteenTwelths = 13.0 / 12.0;
  double eps = 1e-6;

  // Estimate the smoothness of each stencil.

  double s1 = thirteenTwelths * (v1 - 2 * v2 + v3) * (v1 - 2 * v2 + v3)
              + oneQuarter * (v1 - 4 * v2 + 3 * v3) * (v1 - 4 * v2 + 3 * v3);

  double s2 = thirteenTwelths * (v2 - 2 * v3 + v4) * (v2 - 2 * v3 + v4)
              + oneQuarter * (v2 - v4) * (v2 - v4);

  double s3 = thirteenTwelths * (v3 - 2 * v4 + v5) * (v3 - 2 * v4 + v5)
              + oneQuarter * (3 * v3 - 4 * v4 + v5) * (3 * v3 - 4 * v4 + v5);

  // Compute the alpha values for each stencil.

  double alpha1 = 0.1 / ((s1 + eps) * (s1 + eps));
  double alpha2 = 0.6 / ((s2 + eps) * (s2 + eps));
  double alpha3 = 0.3 / ((s3 + eps) * (s3 + eps));

  // Calculate the normalised weights.

  double totalWeight = alpha1 + alpha2 + alpha3;

  double w1 = alpha1 / totalWeight;
  double w2 = alpha2 / totalWeight;
  double w3 = alpha3 / totalWeight;

  // Sum the three stencil components.
  double grad = w1 * (2 * v1 - 7 * v2 + 11 * v3) + w2 * (5 * v3 - v2 + 2 * v4)
                + w3 * (2 * v3 + 5 * v4 - v5);

  grad *= (1.0 / 6.0);

  return grad;
}

double LevelSet::pointToLineDistance (const Coord& vertex1, const Coord& vertex2,
                                      const Coord& point) const {
  // Separation components between vertices.
  double dx = vertex2.x - vertex1.x;
  double dy = vertex2.y - vertex1.y;

  // Squared separation.
  double rSqd = dx * dx + dy * dy;

  // Return separation between point and either vertex.
  if (rSqd < 1e-6) {
    dx = point.x - vertex1.x;
    dy = point.y - vertex1.y;

    return std::sqrt (dx * dx + dy * dy);
  }

  /* Consider the line extending the segment, parameterized as v + t (w - v).
     We find the projection of the point p onto the line. It falls where

       t = [(p-v) . (w-v)] / |w-v|^2

     We clamp t from [0,1] to handle points outside the segment vw, i.e. if
     t < 0 then we use the distance from p to v, and if t > 1 we use the
     distance from p to w.

     (Where v = vertex1, w = vertex2, and p = point)
   */
  else {
    double t = ((point.x - vertex1.x) * dx + (point.y - vertex1.y) * dy) / rSqd;

    t = std::max (0.0, std::min (1.0, t));

    // Project onto the line.
    double x = vertex1.x + t * dx;
    double y = vertex1.y + t * dy;

    // Compute distance from the point.
    dx = x - point.x;
    dy = y - point.y;

    return std::sqrt (dx * dx + dy * dy);
  }
}

bool LevelSet::isInsidePolygon (const Coord& point, const std::vector<Coord>& vertices) const {
  /* Test whether a point lies inside a polygon.

     The polygon is defined by a vector of vertices. These must be ordered
     and closed, i.e. there are n vertices with vertices[n] = vertices[0].

     The test calculates the winding number of the point and returns false
     only when it is zero.

     Adapted from: http://geomalgorithms.com/a03-_inclusion.html
   */

  int windingNumber = 0;

  // Loop through all vertices.
  for (unsigned int i = 0; i < vertices.size () - 1; i++) {
    // Point lies above the vertex.
    if (vertices[i].y <= point.y) {
      // Point lies below the next vertex.
      if (vertices[i + 1].y > point.y)

        // Point is to the left of the edge.
        if (isLeftOfLine (vertices[i], vertices[i + 1], point) > 0) windingNumber++;
    } else {
      // Point lies above the vertex.
      if (vertices[i + 1].y <= point.y)

        // Point lies to the right of the edge.
        if (isLeftOfLine (vertices[i], vertices[i + 1], point) < 0) windingNumber--;
    }
  }

  if (windingNumber == 0)
    return false;
  else
    return true;
}

int LevelSet::isLeftOfLine (const Coord& vertex1, const Coord& vertex2, const Coord& point) const {
  return ((vertex2.x - vertex1.x) * (point.y - vertex1.y)
          - (point.x - vertex1.x) * (vertex2.y - vertex1.y));
}

// Function for sparse fast velocity extension. Useful for mapping the changes of the level set in
// the nodes due to a change in the boundary points (= dy/dx).
Eigen::VectorXd LevelSet::computedydx (bool assemble_A, bool time_it, std::vector<double>& x,
                                       const std::vector<BoundaryPoint> points,
                                       const std::vector<BoundarySegment> segments) {
  /*
      Note: Narrow band identification should be done element-wise i.e. if any nodes on an element
     are in the narrow band, then all of them should be included.
  */

  auto t_start = std::chrono::high_resolution_clock::now ();

  if (time_it) {
    std::cout << "\nSparse fast velocity extension";

    if (assemble_A) {
      std::cout << " (assemble A version)";
    }

    std::cout << " ... " << std::flush;
  }

  int spacedim = 2;

  /*
      The vector x should contain the displacement at the nodes in surface_mesh:
  */

  assert (x.size () == points.size ());

  Eigen::VectorXd f = Eigen::VectorXd::Zero (signedDistance.size ());

  /*
      The matrix [A] maps the velocity of the surface mesh nodes to that of the
      solid mesh nodes (where ϕ is defined):
  */

  std::vector<int> acceptance_order, acceptance_order_far;
  acceptance_order.reserve (signedDistance.size ());

  std::vector<std::vector<double> > dydx (0);
  std::vector<std::vector<int> > dydx_ids (0);

  if (assemble_A) {
    dydx = std::vector<std::vector<double> > (signedDistance.size (), std::vector<double> (2, 0.0));
    dydx_ids = std::vector<std::vector<int> > (signedDistance.size (), std::vector<int> (2, -1));
  }

  /*
      Since fast marching should never change the sign of any
      signedDistance, you can run the whole algorithm in the "positive sense"
      i.e. every signedDistance is given a positive value, then the sign
      is corrected at the end (to be the same as it was before
      the algorithm).

      Doing it like this makes the equations easier to solve,
      branches of quadratics easier to choose, etc.
  */

  // Eigen::VectorXd signedDistance_new = signedDistance.array().abs() ;
  std::vector<double> signedDistance_new = signedDistance;

  /*
      The first step is to calculate the distance from the level set boundary
      to the nearest adjacent nodes, and tag them as "accepted" (note in some
      papers, these nodes are tagged as "alive").
  */

  std::vector<bool> signedDistance_accepted (signedDistance.size (), false);
  int n_accepted = 0;

  /*
      Now use the segment elements:
  */

  // for (auto && segment : boundary.segments)
  for (int i = 0; i < segments.size (); i++) {
    // Nodes of the element where the segment is in.
    std::vector<int> node_ids;
    node_ids.resize (4);
    node_ids[0] = mesh.elements[segments[i].element].nodes[0];
    node_ids[1] = mesh.elements[segments[i].element].nodes[1];
    node_ids[2] = mesh.elements[segments[i].element].nodes[2];
    node_ids[3] = mesh.elements[segments[i].element].nodes[3];

    // std::cout << "element nodes: " << node_ids[0] << ", " << node_ids[1] << ", " << node_ids[2]
    // << ", " << node_ids[3] << "\n"; std::cout << "segment coord: (" <<
    // points[segments[i].start].coord.x << ", " << points[segments[i].start].coord.y << ") - (" <<
    // points[segments[i].end].coord.x << ", " << points[segments[i].end].coord.y << ")\n";

    // for (auto && id : node_ids)
    for (int j = 0; j < node_ids.size (); j++) {
      std::vector<double> d_f_t = pointToLineDistanceAndValue (node_ids[j], points, segments, i, x);
      // std::cout << "d_f_t: " << d_f_t[0] << "\n";

      if (not signedDistance_accepted[node_ids[j]]) {
        signedDistance_new[node_ids[j]] = d_f_t[0];
        f[node_ids[j]] = d_f_t[1];
        signedDistance_accepted[node_ids[j]] = true;
        n_accepted += 1;
        acceptance_order.push_back (node_ids[j]);

        if (assemble_A) {
          dydx[node_ids[j]][0] = 1.0 - d_f_t[2];
          dydx[node_ids[j]][1] = d_f_t[2];

          dydx_ids[node_ids[j]][0] = segments[i].start;
          dydx_ids[node_ids[j]][1] = segments[i].end;
        }

      }  // if not accepted.

      else if (d_f_t[0] < signedDistance_new[node_ids[j]]) {
        signedDistance_new[node_ids[j]] = d_f_t[0];
        f[node_ids[j]] = d_f_t[1];

        if (assemble_A) {
          dydx[node_ids[j]][0] = 1.0 - d_f_t[2];
          dydx[node_ids[j]][1] = d_f_t[2];

          dydx_ids[node_ids[j]][0] = segments[i].start;
          dydx_ids[node_ids[j]][1] = segments[i].end;
        }
      }

    }  // for node_ids.

  }  // for boundary.segments.

  if (!(n_accepted == signedDistance.size ())) {
    acceptance_order_far.reserve (signedDistance.size () - acceptance_order.size ());

    /*
        Now all the grid points adjacent to the interface should be initialized
        with the exact values of signed distance.

        From Osher & Fedkiw's Yellow Book pg. 70:

        In order to march out from this initial band, constructing the distance
        function as we go, we need to decide which grid point to update first.
        This should be the one that the zero isocontour would cross first in the
        crossing-time method, i.e., the grid point with the smallest crossing time
        or smallest value of distance. So, for each grid point adjacent to the band,
        we calculate a tentative value for the distance function. This is done
        using only the values of signedDistance that have already been accepted into the band;
        i.e., tentative values do not use other tentative values in this calculation.
        Then we choose the point with the smallest tentative value to add to the band
        of accepted grid points. Since the signed distance function is created with
        characteristics that flow out of the interface in the normal direction, this
        chosen point does not depend on any of the other tentative grid points that
        will have larger values of distance. Thus, the tentative value of distance
        assigned to this grid point is an acceptable approximation of the signed
        distance function.

        Now that the band of accepted values has been increased by one, we repeat the
        process. Most of the grid points in the tentative band already have good
        tentative approximations to the distance function. Only those adjacent to the
        newly added point need modification. Adjacent tentative grid points need their
        tentative values updated using the new information gained by adding the chosen
        point to the band. Any other adjacent grid point that did not yet have a
        tentative value needs to have a tentative value assigned to it using the values
        in the band of accepted points. Then we choose the smallest tentative value, add
        it to the band, and repeat the algorithm. Eventually, every grid point in Ω+ gets
        added to the band, completing the process. As noted above, the grid points in Ω−
        are updated with a similar process.

        The slowest part of this algorithm is the search through all the tentative grid
        points to find the one with the smallest value. This search can be accelerated
        using a binary tree to store all the tentative values. The tree is organized so that
        each point has a tentative distance value smaller than the two points located
        below it in the tree. This means that the smallest tentative point is always
        conveniently located at the top of the tree.
    */

    /*
        We use the multimap class for doing the fast marching,
        namely storing the close points and tentative signed
        distance values.

        It is guaranteed to be as fast as a binary tree.

        The (key, value) structure we use is:

            key   --> the distance to the boundary
            value --> the index of the node (and the signedDistance vector)

        To insert into the map (which we call "tree") use, e.g:
        tree[1.234] = 5 ;

        To erase the first (lowest key = distance value) element use:
        tree.erase(tree.begin()) ;

        In case you need to access the first element use:
        tree.begin()->first for the key, tree.begin()->second for the value.

        In case you need to access the last element use:
        tree.rbegin()->first for the key, tree.rbegin()->second for the value.
    */

    std::multimap<double, int> tree;

    /*
        On the first iteration, we have to loop through all the
        accepted points; subsequent iterations need only attend to the
        most recently accepted point.
    */

    for (int j = 0; j < signedDistance.size (); ++j) {
      if (signedDistance_accepted[j]) {
        /*
            Update the tentative values for all the
            neighbours.
        */

        // Elements which the the node is connected.
        std::vector<int> neighbour_ids;
        neighbour_ids.resize (4);
        neighbour_ids[0] = mesh.nodes[j].neighbours[0];
        neighbour_ids[1] = mesh.nodes[j].neighbours[1];
        neighbour_ids[2] = mesh.nodes[j].neighbours[2];
        neighbour_ids[3] = mesh.nodes[j].neighbours[3];

        for (int k = 0; k < neighbour_ids.size (); ++k) {
          /*
              Note that not all nodes have neighbours on the left
              and right in all dimensions. E.g. if a node is on the
              left edge, it's left neighbour_id will be nNodes.

              So check to make sure the neighbour_id is != nNodes.
          */

          if (neighbour_ids[k] != mesh.nNodes and not signedDistance_accepted[neighbour_ids[k]]) {
            std::vector<int> signedDistance_s_id;
            std::vector<double> b;
            updateValue (neighbour_ids[k], signedDistance_new, signedDistance_accepted, f,
                         signedDistance_s_id, b);

            if (assemble_A) {
              for (int l = 0; l < spacedim; ++l) {
                dydx[neighbour_ids[k]][l] = b[l];
                dydx_ids[neighbour_ids[k]][l] = signedDistance_s_id[l];

              }  // for l (signedDistance_s_id).

            }  // if assemble_A.

            /*
                Push this neighbour_id into the "close" tree.
                I only use the absolute distance value to make
                pulling the smallest distance id easily; the
                signed-distance is stored in signedDistance anyway.
            */

            tree.insert (std::make_pair (signedDistance_new[neighbour_ids[k]], neighbour_ids[k]));

          }  // if neighbour_ids[k] >= 0 and not already accepted.

        }  // for k (neighbours).

      }  // if signedDistance_accepted[j].

    }  // for j (nodes).

    /*
        On the first iteration, we have to loop through all the
        accepted points; subsequent iterations need only attend to the
        most recently accepted point.
    */

    int newly_accepted = -1;

    while (n_accepted < signedDistance.size ()) {
      /*
          Accept the lowest-distance tentative point (in tree).

          Note that, due to our use of the multimap, we can have
          double entries, so we need to check if the top entry in
          the tree has already been accepted; if so, try the second
          and so on.
      */

      while (true) {
        newly_accepted = tree.begin ()->second;
        tree.erase (tree.begin ());

        if (!signedDistance_accepted[newly_accepted]) {
          signedDistance_accepted[newly_accepted] = true;
          n_accepted += 1;
          acceptance_order.push_back (newly_accepted);
          acceptance_order_far.push_back (newly_accepted);
          break;
        }
      }

      /*
          Update the tentative values for all the
          neighbours of the newly-accepted point:
      */

      std::vector<int> neighbour_ids;
      neighbour_ids.resize (4);
      neighbour_ids[0] = mesh.nodes[newly_accepted].neighbours[0];
      neighbour_ids[1] = mesh.nodes[newly_accepted].neighbours[1];
      neighbour_ids[2] = mesh.nodes[newly_accepted].neighbours[2];
      neighbour_ids[3] = mesh.nodes[newly_accepted].neighbours[3];

      for (int k = 0; k < neighbour_ids.size (); ++k) {
        /*
            Note that not all nodes have neighbours on the left
            and right in all dimensions. E.g. if a node is on the
            left edge, it's left neighbour_id will be nNodes.

            So check to make sure the neighbour_id is != nNodes.
        */

        if (neighbour_ids[k] != mesh.nNodes and not signedDistance_accepted[neighbour_ids[k]]) {
          std::vector<int> signedDistance_s_id;
          std::vector<double> b;
          updateValue (neighbour_ids[k], signedDistance_new, signedDistance_accepted, f,
                       signedDistance_s_id, b);

          if (assemble_A) {
            for (int l = 0; l < spacedim; ++l) {
              dydx[neighbour_ids[k]][l] = b[l];
              dydx_ids[neighbour_ids[k]][l] = signedDistance_s_id[l];

            }  // for l (signedDistance_s_id).

          }  // if assemble_A.

          /*
              Push this neighbour_id into the "close" tree.
              I only use the absolute distance value to make
              pulling the smallest distance id easily; the
              signed-distance is stored in signedDistance anyway.
          */

          tree.insert (std::make_pair (signedDistance_new[neighbour_ids[k]], neighbour_ids[k]));

        }  // if neighbour_ids[k] >= 0 and not already accepted.

      }  // for k (neighbours).

    }  // while there are still unaccepted points.

    if (assemble_A) {
      for (int id1 : acceptance_order_far) {
        std::vector<double> t1 (x.size (), 0.0);

        int n_dydx = 0;

        for (int id2 : dydx_ids[id1]) {
          if (id2 >= 0) {
            n_dydx += dydx_ids[id2].size ();
          }
        }

        std::vector<double> dydx_final;
        std::vector<int> dydx_ids_final;

        dydx_final.reserve (n_dydx);
        dydx_ids_final.reserve (n_dydx);

        int count = 0;

        for (int i = 0; i < dydx_ids[id1].size (); ++i) {
          int id2 = dydx_ids[id1][i];

          if (id2 >= 0) {
            for (int j = 0; j < dydx_ids[id2].size (); ++j) {
              int id3 = dydx_ids[id2][j];
              t1[id3] += dydx[id1][i] * dydx[id2][j];

            }  // for j.

          }  // if id >= 0.

        }  // for i.

        for (int i = 0; i < t1.size (); ++i) {
          if (t1[i] > 0.0) {
            dydx_final.push_back (t1[i]);
            dydx_ids_final.push_back (i);
          }

        }  // for i.

        dydx[id1] = dydx_final;
        dydx_ids[id1] = dydx_ids_final;

      }  // for id1.

    }  // if assemble_A.

  }  // if (!(n_accepted == signedDistance.size())).

  /*
      Assemble [A_sparse]:
  */

  if (assemble_A) {
    A_sparse.resize (0, 0);

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;

    triplet_list.reserve (signedDistance.size () * x.size ());  // Way oversized.

    for (int i = 0; i < dydx_ids.size (); ++i) {
      for (int j = 0; j < dydx_ids[i].size (); ++j) {
        triplet_list.push_back (T (i, dydx_ids[i][j], dydx[i][j]));

      }  // for j.

    }  // for i.

    A_sparse.resize (signedDistance.size (), x.size ());
    A_sparse.setFromTriplets (triplet_list.begin (), triplet_list.end ());

  }  // if assemble_A.

  /*
      Aaaaaaand we're done.
  */

  auto t_end = std::chrono::high_resolution_clock::now ();

  if (time_it) {
    std::cout << "Done. Time elapsed = " << std::chrono::duration<double> (t_end - t_start).count ()
              << "\n";
  }

  return f;
}

std::vector<double> LevelSet::pointToLineDistanceAndValue (
    int node_id, const std::vector<BoundaryPoint> points,
    const std::vector<BoundarySegment> segments, int segment_id, std::vector<double>& x) {
  std::vector<double> d_f_t (3, 0.0);

  std::vector<int> element_node_ids;
  element_node_ids.push_back (segments[segment_id].start);
  element_node_ids.push_back (segments[segment_id].end);

  Eigen::VectorXd v = Eigen::VectorXd::Zero (2);
  v[0] = points[element_node_ids[0]].coord.x;
  v[1] = points[element_node_ids[0]].coord.y;
  Eigen::VectorXd w = Eigen::VectorXd::Zero (2);
  w[0] = points[element_node_ids[1]].coord.x;
  w[1] = points[element_node_ids[1]].coord.y;
  Eigen::VectorXd p = Eigen::VectorXd::Zero (2);
  p[0] = mesh.nodes[node_id].coord.x;
  p[1] = mesh.nodes[node_id].coord.y;

  double l = (v - w).norm ();

  if (l == 0) {
    d_f_t[0] = (p - v).norm ();
    d_f_t[1] = x[element_node_ids[0]];
    d_f_t[2] = 0.0;

  }

  else {
    double t = std::max (0.0, std::min (1.0, (p - v).dot (w - v) / (l * l)));
    Eigen::VectorXd u = v + t * (w - v);
    d_f_t[0] = (p - u).norm ();
    d_f_t[1] = x[element_node_ids[0]] + (x[element_node_ids[1]] - x[element_node_ids[0]]) * t;
    d_f_t[2] = t;
  }

  return d_f_t;
}

void LevelSet::updateValue (int id, std::vector<double>& phi_new, std::vector<bool>& phi_accepted,
                            Eigen::VectorXd& f, std::vector<int>& phi_s_id,
                            std::vector<double>& b) {
  /*
      This function returns a seemingly-useless vector,
      phi_s_id, which stores the id of the nodes used
      to update the phi value.

      This phi_s_id is not needed when doing fast marching,
      though it is needed for the velocity extension
      algorithm.

      First we need the neighbour ids for this node, which
      we will use to calculate signed-distance:
  */

  std::vector<int> neighbour_ids;
  neighbour_ids.resize (4);
  neighbour_ids[0] = mesh.nodes[id].neighbours[0];
  neighbour_ids[1] = mesh.nodes[id].neighbours[1];
  neighbour_ids[2] = mesh.nodes[id].neighbours[2];
  neighbour_ids[3] = mesh.nodes[id].neighbours[3];

  /*
      The crux of this part of the algorithm is to solve phi from a
      quadratic equation of the form:

      sum ( (phi - phi_s)/dx )^2 = 1

      (see the Yellow book, pg. 72). There is a bit of logic involved
      as numeric errors can cause imaginary solutions, so we will
      pass the requisite information to another function. Such information
      is phi_s and dx.
  */

  int spacedim = 2;

  std::vector<double> phi_s (spacedim, 0.0);
  std::vector<double> dx (spacedim, 0.0);
  std::vector<bool> phi_s_accepted (spacedim, false);
  // std::vector<int> phi_s_id (spacedim, 0.0) ;
  phi_s_id = std::vector<int> (spacedim, 0.0);

  for (int l = 0; l < spacedim; ++l) {
    /*
        If there is a left-side neighbour and this is accepted:
    */

    if (neighbour_ids[2 * l] != mesh.nNodes and phi_accepted[neighbour_ids[2 * l]]) {
      phi_s_accepted[l] = true;
      if (l == 0) {
        dx[l] = mesh.nodes[id].coord.x - mesh.nodes[neighbour_ids[2 * l]].coord.x;
      } else if (l == 1) {
        dx[l] = mesh.nodes[id].coord.y - mesh.nodes[neighbour_ids[2 * l]].coord.y;
      } else {
        std::cout << "\n\nERROR in update value due to dimension!\n\n";
      }
      phi_s[l] = phi_new[neighbour_ids[2 * l]];
      phi_s_id[l] = neighbour_ids[2 * l];
    }

    /*
        If there is a right-side neighbour and this is accepted:
    */

    if (neighbour_ids[2 * l + 1] != mesh.nNodes and phi_accepted[neighbour_ids[2 * l + 1]]) {
      if (not phi_s_accepted[l]
          or phi_new[neighbour_ids[2 * l + 1]] < phi_new[neighbour_ids[2 * l]]) {
        phi_s_accepted[l] = true;
        if (l == 0) {
          dx[l] = mesh.nodes[neighbour_ids[2 * l + 1]].coord.x - mesh.nodes[id].coord.x;
        } else if (l == 1) {
          dx[l] = mesh.nodes[neighbour_ids[2 * l + 1]].coord.y - mesh.nodes[id].coord.y;
        } else {
          std::cout << "\n\nERROR in update value due to dimension!\n\n";
        }
        phi_s[l] = phi_new[neighbour_ids[2 * l + 1]];
        phi_s_id[l] = neighbour_ids[2 * l + 1];
      }
    }

  }  // for l (spacedim).

  /*
      Solve the quadratic equation:
  */

  phi_new[id] = solveQuadratic (phi_s, dx, phi_s_accepted);

  /*
      Now based on the solve_quadratic algorithm, the values
      in phi_s_accepted may have changed (numerical innaccuracies
      require the rejection of some points). So we update
      phi_s_id accordingly here:
  */

  for (int l = 0; l < spacedim; ++l) {
    if (not phi_s_accepted[l]) {
      phi_s_id[l] = -1;
    }
  }

  /*
      Velocity extension:

          ∇F * ∇ϕ = 0
          Σ (f - fs)/Δx * (ϕ - ϕs)/Δx = 0

      Solve for f as:

          den = Σ (ϕ - ϕs)/(Δx * Δx)
            f = {(ϕ - ϕs)/(Δx * Δx)/den}^T * {fs}
              ≡ {b}^T * {fs}
  */

  // double num = 0.0, den = 0.0 ;
  double den = 0.0;
  std::vector<double> fs (spacedim, 0.0);
  b = std::vector<double> (spacedim, 0.0);

  for (int l = 0; l < spacedim; ++l) {
    if (phi_s_id[l] >= 0) {
      // den += 1.0 / dx[l] * (phi_new[id] - phi_new[phi_s_id[l]]) / dx[l] ;
      // num += f[phi_s_id[l]] / dx[l] * (phi_new[id] - phi_new[phi_s_id[l]]) / dx[l] ;

      den += (phi_new[id] - phi_new[phi_s_id[l]]) / (dx[l] * dx[l]);
      b[l] = (phi_new[id] - phi_new[phi_s_id[l]]) / (dx[l] * dx[l]);
      fs[l] = f[phi_s_id[l]];
    }
  }

  for (int l = 0; l < spacedim; ++l) {
    b[l] /= den;
  }

  // f[id] = num / den ;

  f[id] = vector_dot (b, fs);
}

double LevelSet::solveQuadratic (std::vector<double>& phi_s, std::vector<double>& dx,
                                 std::vector<bool>& phi_s_accepted) {
  /*
      Solves the following equation for phi:

          sum ( (phi - phi_s)/dx )^2 = 1

      where there is one phi_s for each spatial dimension. Note that
      not all spacial dimensions are accepted; the ones that aren't
      won't be used here. See yellow book pg. 72.
  */

  /*
      First we define

          p(x) = sum ( (x - phi_s)/dx )^2

      Now in order to satisfy the quadratic equation, a necessary
      and sufficient condition is

          p(max(phi_s)) <= 1

      If p(max(phi_s)) > 1, a solution strategy is to remove
      the max(phi_s) value, and try again.
  */

  int spacedim = 2;

  while (true) {
    double max_id = -1;

    for (int i = 0; i < spacedim; ++i) {
      if (phi_s_accepted[i] && (max_id < 0 or phi_s[i] > phi_s[max_id])) {
        max_id = i;
      }
    }

    double p = 0;

    for (int i = 0; i < spacedim; ++i) {
      if (phi_s_accepted[i]) {
        p += std::pow ((phi_s[max_id] - phi_s[i]) / dx[i], 2.0);
      }
    }

    if (p <= 1) {
      break;

    }

    else {
      phi_s_accepted[max_id] = false;
    }
  }

  /*
      So now it should be possible to solve the quadratic equation:
  */

  double a = 0, b = 0, c = -1;

  for (int i = 0; i < spacedim; ++i) {
    if (phi_s_accepted[i]) {
      a += 1.0 / std::pow (dx[i], 2.0);
      b += -2.0 * phi_s[i] / std::pow (dx[i], 2.0);
      c += std::pow (phi_s[i] / dx[i], 2.0);
    }
  }

  double discrim = b * b - 4 * a * c;

  if (discrim < 0) {
    std::cout
        << "\n\n*****\n\nError in solve_quadratic() in fast_march(): discriminant < 0\n\n*****\n\n";
    throw;
  }

  return (-b + std::sqrt (discrim)) / (2 * a);
}

double LevelSet::vector_dot (std::vector<double>& a, std::vector<double>& b) {
  if (a.size () != b.size ()) {
    throw std::invalid_argument (
        "\n\n*****\nInput vector dimension problem in M2DO::FEA vector_dot ().\n*****\n\n");
  }

  double dot = 0.0;

  for (int i = 0; i < a.size (); ++i) {
    dot += a[i] * b[i];
  }

  return dot;
}

}  // namespace lsm_2d