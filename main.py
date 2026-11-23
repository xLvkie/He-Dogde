"""Entry point for 'Esquiva la Silueta'."""

from __future__ import annotations

import argparse
import sys

import pygame

from config import (
    GAME_DURATION_SECONDS,
    SILHOUETTE_SCALE,
    VIRTUAL_H,
    VIRTUAL_W,
)
from game import EsquivaLaSiluetaGame
from segmentation import SilhouetteSegmenter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esquiva la Silueta: videojuego arcade controlado con tu silueta."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de la cámara web OpenCV (por defecto: 0).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecuta en modo demo con silueta sintética controlable sin cámara web.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=GAME_DURATION_SECONDS,
        help="Duración objetivo de la partida en segundos (por defecto: 300.0 = 5 min).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=SILHOUETTE_SCALE,
        help=f"Factor de escala de la silueta en pantalla (por defecto: {SILHOUETTE_SCALE}).",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Inicia en modo ventana en lugar de pantalla completa nativa.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=VIRTUAL_W,
        help=f"Ancho virtual del área de juego (por defecto: {VIRTUAL_W}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=VIRTUAL_H,
        help=f"Alto virtual del área de juego (por defecto: {VIRTUAL_H}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    segmenter = SilhouetteSegmenter(
        camera_index=args.camera,
        display_size=(args.width, args.height),
        silhouette_scale=args.scale,
        demo_mode=args.demo,
    )
    try:
        game = EsquivaLaSiluetaGame(
            segmenter=segmenter,
            duration_seconds=args.duration,
            width=args.width,
            height=args.height,
            fullscreen=not args.windowed,
        )
        game.run()
    finally:
        segmenter.close()
        if pygame.get_init():
            pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
