import pygame

from typing import Tuple


class PygameWindow:
  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode((600, 600))
    self._clock = pygame.time.Clock()
    self._font = pygame.font.SysFont(None, 30)

  def render_text(self, text: str, position: Tuple[int, int]):
    pygame_text = self._font.render(text, True, pygame.Color('black'))
    self.screen.blit(pygame_text, position)

  def finish_render(self):
    pygame.display.flip()
    self._clock.tick(30)

  def render_clear(self):
    self.screen.fill(pygame.Color('white'))

  def render_stats(self, total: int, wins: int, running: bool):
    if total > 0:
      self.render_text('wins%: {:.2f}'.format(wins / total * 100), (0, 40))
    self.render_text(f'total: {total}', (0, 60))
    if not running:
      self.render_text('Preparing to shut down...', (200, 40))

  @staticmethod
  def check_close_event() -> bool:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
    return True

