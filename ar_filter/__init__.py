"""
AR Filter Module - Snapchat-style face filter using OpenGL + MediaPipe

This module is completely independent from the main project.
It provides a real-time face filter with neon mask effects.

Usage:
    from ar_filter.gl_app import run_ar_filter
    run_ar_filter()
"""

__version__ = "1.0.0"
__author__ = "AR Filter Module"

from .gl_app import run_ar_filter

__all__ = ["run_ar_filter"]
