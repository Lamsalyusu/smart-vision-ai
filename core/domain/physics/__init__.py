"""3D Rigid Body Physics, Contact Impulses, and Spatial Dynamics Simulation Subsystem."""

from .collision import GJKCollisionDetector
from .contact_solver import ContactImpulseSolver
from .integrator import NumericalPhysicsIntegrator
from .rigid_body import RigidBody3D

__all__ = [
    "ContactImpulseSolver",
    "GJKCollisionDetector",
    "NumericalPhysicsIntegrator",
    "RigidBody3D",
]
