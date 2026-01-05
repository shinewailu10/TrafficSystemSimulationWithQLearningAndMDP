import numpy as np
import pygame
import time
import random
import os

#Constents
CAR_WIDTH = 34    
CAR_LENGTH = 64   
GAP_SIZE = 110    
WAIT_THRESHOLD = 300 

class AssetManager:
    def __init__(self):
        self.images = {}
        self.car_images = [] 
        self.load_assets()

    def load_assets(self):
        if not os.path.exists("assets"):
            os.makedirs("assets")
            print("Created 'assets' folder.")

        for i in range(1, 8): 
            filename = f"assets/car{i}.png"
            try:
                raw_car = pygame.image.load(filename)
                raw_car = pygame.transform.scale(raw_car, (CAR_WIDTH, CAR_LENGTH))
                self.car_images.append(raw_car)
            except:
                self.car_images.append(None)

        try:
            grass = pygame.image.load("assets/grass.jpg")
            grass = pygame.transform.scale(grass, (800, 600))
            self.images['grass'] = grass
        except:
            self.images['grass'] = None

        try:
            road = pygame.image.load("assets/road.jpg")
            self.images['road'] = road
        except:
            self.images['road'] = None

        try:
            building = pygame.image.load("assets/building.png")
            self.images['building'] = building
        except:
            self.images['building'] = None

        try:
            tl = pygame.image.load("assets/traffic_light.png")
            tl = pygame.transform.scale(tl, (40, 100)) 
            self.images['traffic_light'] = tl
        except:
            self.images['traffic_light'] = None

    def get_car_image(self, direction, car_index):
        if not self.car_images or car_index >= len(self.car_images) or self.car_images[car_index] is None:
            return None
        img = self.car_images[car_index].copy()
        if direction == 'up': rotated = img 
        elif direction == 'down': rotated = pygame.transform.rotate(img, 180)
        elif direction == 'left': rotated = pygame.transform.rotate(img, 90)
        elif direction == 'right': rotated = pygame.transform.rotate(img, -90)
        return rotated

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(2, 5)
        self.life = 255
        self.drift_x = random.uniform(-0.5, 0.5)
        self.drift_y = random.uniform(-0.5, 0.5)

    def update(self):
        self.x += self.drift_x
        self.y += self.drift_y
        self.size += 0.1 
        self.life -= 10 

class CarEntity:
    def __init__(self, lane, stop_pos, start_pos, direction, sprite_index):
        self.lane = lane
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.direction = direction
        self.sprite_index = sprite_index
        self.color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        self.speed = 0
        self.max_speed = 4.0    
        self.accel = 0.2        
        self.decel = 0.2        
        self.state = "approaching" 
        self.stop_pos_base = stop_pos 
        self.is_braking = False
        self.wait_time = 0

    def update(self, car_index_in_queue, gap_spacing=GAP_SIZE): 
        if self.state == "waiting" and self.speed < 0.1:
            self.wait_time += 1
        
        if self.state == "leaving":
            self.speed = min(self.speed + 0.2, 6.0) 
            self.is_braking = False
            self.move_by_speed() 
            return

        if self.direction == 'down':
            target = self.stop_pos_base - (car_index_in_queue * gap_spacing)
            dist = target - self.y
        elif self.direction == 'up':
            target = self.stop_pos_base + (car_index_in_queue * gap_spacing)
            dist = self.y - target
        elif self.direction == 'left':
            target = self.stop_pos_base + (car_index_in_queue * gap_spacing)
            dist = self.x - target
        elif self.direction == 'right':
            target = self.stop_pos_base - (car_index_in_queue * gap_spacing)
            dist = target - self.x

        if dist > 30: 
            self.speed = min(self.speed + self.accel, self.max_speed)
            self.is_braking = False
            self.state = "approaching"
        elif dist > 1: 
            self.speed = max(0.0, dist * 0.15) 
            self.is_braking = True 
        else: 
            self.speed = 0
            self.is_braking = True 
            self.state = "waiting"

        if self.speed > 0:
            self.move_by_speed() 

    def move_by_speed(self):
        if self.direction == 'down': self.y += self.speed
        elif self.direction == 'up': self.y -= self.speed
        elif self.direction == 'left': self.x -= self.speed
        elif self.direction == 'right': self.x += self.speed

class TrafficEnv:
    def __init__(self, visualizer=None):
        self.action_space = [0, 1, 2, 3] 
        self.visualizer = visualizer
        self.current_green = 0 
        self.steps_in_current_phase = 0
        self.min_duration = 40  
        self.max_green_duration = 60 
        self.reset()

    @property
    def state(self):
        if self.visualizer:
            return np.array([len(l) for l in self.visualizer.lanes])
        return np.zeros(4, dtype=int)

    def _get_simplified_state(self):
        return tuple(min(c, 20) for c in self.state)

    def reset(self):
        if self.visualizer:
            self.visualizer.reset_cars()
            for i in range(4):
                count = np.random.randint(1, 4) 
                for _ in range(count):
                    self.visualizer.add_car(i, instant=True)
        self.current_green = 0 
        self.steps_in_current_phase = 0
        self.prev_wait = 0
        return self._get_simplified_state()

    def step(self, action):
        if self.steps_in_current_phase < self.min_duration:
            action = self.current_green

        if self.steps_in_current_phase > self.max_green_duration:
            action = np.argmax(self.state)

        switched = (action != self.current_green)

        if switched:
            if self.visualizer and pygame.get_init():
                while True:
                    if self.visualizer.is_intersection_clear():
                        break
                    self.visualizer.update_physics(green_lane=-1)
                    if not self.visualizer.draw(active_index=-1):
                        return None, 0
                    pygame.event.pump()

            self.current_green = action
            self.steps_in_current_phase = 0
        else:
            self.steps_in_current_phase += 1

        self._random_arrivals()

        #reward system
        total_queue = sum(self.state)          
        queue_penalty = - (total_queue / 20.0)   

        wait_penalty = 0.0
        if self.visualizer:
            prev_wait = getattr(self, "prev_wait", 0)
            curr_wait = sum(car.wait_time for lane in self.visualizer.lanes for car in lane)
            wait_penalty = (prev_wait - curr_wait) / 50.0
            self.prev_wait = curr_wait

        flow_bonus = 0.2 if self.state[self.current_green] > 0 else 0.0

        change_penalty = -0.2 if action != self.current_green else 0.0

        reward = queue_penalty + wait_penalty + flow_bonus + change_penalty

        reward = max(-1.0, min(1.0, reward))

        return self._get_simplified_state(), reward


    def _random_arrivals(self):
        for i in range(4):
            if np.random.random() < 0.20: 
                if len(self.visualizer.lanes[i]) < 20: 
                    self.visualizer.add_car(i)

    def render(self, action):
        if self.visualizer and pygame.get_init():
            self.visualizer.update_physics(green_lane=action)
            return self.visualizer.draw(active_index=action)
        return False

class TrafficVisualizer:
    def __init__(self):
        pygame.init()
        self.screen_size = (800, 600)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Master's Project: Traffic AI")
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.queue_font = pygame.font.SysFont("Arial", 24, bold=True)

        self.cx, self.cy = 400, 300
        self.road_w = 140 
        self.lane_w = self.road_w // 2
        self.stop_off = self.road_w // 2 + 15

        self.assets = AssetManager()
        self.particles = [] 
        self.running = True

        self.colors = {
            'line_white': (220, 220, 220), 'line_yellow': (220, 180, 20),
            'sidewalk': (160, 160, 160), 
            'pole': (40, 40, 40), 
            'window': (100, 200, 255), 'window_off': (50, 50, 80)
        }
        
        self.trees = []
        zones = [
            (10, 10, self.cx - self.road_w, self.cy - self.road_w), 
            (self.cx + self.road_w, 10, 800, self.cy - self.road_w), 
            (0, self.cy + self.road_w, self.cx - self.road_w, 600), 
            (self.cx + self.road_w, self.cy + self.road_w, 800, 600) 
        ]
        
        for zone in zones:
            x1, y1, x2, y2 = zone
            for _ in range(15):
                tx = random.randint(int(x1) + 20, int(x2) - 20)
                ty = random.randint(int(y1) + 20, int(y2) - 20)
                self.trees.append((tx, ty))
        
        self.lanes = [[], [], [], []] 
        self.leaving_cars = []
        self.last_release_time = 0

    def reset_cars(self):
        self.lanes = [[], [], [], []]
        self.leaving_cars = []
        self.particles = []

    def add_car(self, lane_index, instant=False):
        sprite_index = random.randint(0, 6) 
        offset = 12 
        
        if lane_index == 0: 
            stop, start, d = self.cy - self.stop_off - 80, (self.cx - self.lane_w//2 - offset, -100), 'down'
        elif lane_index == 1: 
            stop, start, d = self.cy + self.stop_off + 20, (self.cx + self.lane_w//2 - offset, 700), 'up'
        elif lane_index == 2: 
            stop, start, d = self.cx + self.stop_off + 20, (900, self.cy - self.lane_w//2 - offset), 'left'
        elif lane_index == 3: 
            stop, start, d = self.cx - self.stop_off - 80, (-100, self.cy + self.lane_w//2 - offset), 'right'

        if not instant and len(self.lanes[lane_index]) > 0:
            last_car = self.lanes[lane_index][-1]
            safe_dist = 110 
            if d == 'down' and last_car.y < start[1] + safe_dist: return
            if d == 'up' and last_car.y > start[1] - safe_dist: return
            if d == 'left' and last_car.x > start[0] - safe_dist: return
            if d == 'right' and last_car.x < start[0] + safe_dist: return

        car = CarEntity(lane_index, stop, start, d, sprite_index)
        if instant: 
            if d=='down': car.y = stop - (len(self.lanes[lane_index]) * GAP_SIZE)
            elif d=='up': car.y = stop + (len(self.lanes[lane_index]) * GAP_SIZE)
            elif d=='left': car.x = stop + (len(self.lanes[lane_index]) * GAP_SIZE)
            elif d=='right': car.x = stop - (len(self.lanes[lane_index]) * GAP_SIZE)
            car.state = "waiting"
        self.lanes[lane_index].append(car)

    def release_car(self, lane_index):
        if len(self.lanes[lane_index]) > 0:
            car = self.lanes[lane_index].pop(0)
            car.state = "leaving"
            self.leaving_cars.append(car)
            return True
        return False

    def is_intersection_clear(self):
        margin = 40 
        x1 = self.cx - self.road_w//2 - margin
        x2 = self.cx + self.road_w//2 + margin
        y1 = self.cy - self.road_w//2 - margin
        y2 = self.cy + self.road_w//2 + margin
        
        for car in self.leaving_cars:
            if x1 < car.x < x2 and y1 < car.y < y2:
                return False
        return True

    def update_physics(self, green_lane):
        cars_released = 0
        current_time = time.time()

        for lane_idx, queue in enumerate(self.lanes):
            for i, car in enumerate(queue):
                car.update(i) 
            
            if lane_idx == green_lane and len(queue) > 0:
                if not self.is_intersection_clear():
                    continue 

                first_car = queue[0]
                if current_time - self.last_release_time > 0.8:
                    if first_car.state == "waiting" or first_car.is_braking:
                        car = queue.pop(0)
                        car.state = "leaving"
                        self.leaving_cars.append(car)
                        self.last_release_time = current_time
                        cars_released += 1

        for car in self.leaving_cars[:]:
            car.update(0) 
            if random.random() < 0.4: self.create_exhaust(car)
            if not (-300 < car.x < 1100 and -300 < car.y < 900):
                self.leaving_cars.remove(car)

        for p in self.particles[:]:
            p.update()
            if p.life <= 0: self.particles.remove(p)
            
        return cars_released

    def create_exhaust(self, car):
        ex_x, ex_y = car.x + CAR_WIDTH//2, car.y + CAR_LENGTH//2
        if car.direction == 'up': ex_y = car.y + CAR_LENGTH
        elif car.direction == 'down': ex_y = car.y
        elif car.direction == 'left': ex_x = car.x + CAR_LENGTH
        elif car.direction == 'right': ex_x = car.x
        self.particles.append(Particle(ex_x, ex_y))

    def draw_cube(self, x, y, w, h, d, top_col, side_col):
        pygame.draw.rect(self.screen, side_col, (x, y + h, w, d)) 
        pygame.draw.rect(self.screen, side_col, (x + w, y, d, h + d)) 
        pygame.draw.rect(self.screen, top_col, (x, y, w, h)) 
        pygame.draw.rect(self.screen, (255,255,255), (x, y, w, h), 1)

    def draw_tree(self, x, y):
        pygame.draw.ellipse(self.screen, (30, 90, 30), (x-15, y+35, 50, 20))
        pygame.draw.rect(self.screen, (100, 60, 40), (x, y, 16, 45))
        leaf_col = (30, 120, 30)
        pygame.draw.circle(self.screen, leaf_col, (x+8, y-10), 22)
        pygame.draw.circle(self.screen, leaf_col, (x-5, y+10), 18)
        pygame.draw.circle(self.screen, leaf_col, (x+20, y+10), 18)
        pygame.draw.circle(self.screen, (50, 160, 50), (x+5, y-12), 15)

    def draw_scenery(self):
        cx, cy, rw = self.cx, self.cy, self.road_w
        if self.assets.images['grass']:
            for x in range(0, 800, 800): 
                self.screen.blit(self.assets.images['grass'], (x, 0))
        else:
            self.screen.fill((50, 160, 50))
        sw_w = rw + 40
        pygame.draw.rect(self.screen, self.colors['sidewalk'], (0, cy - sw_w//2, 800, sw_w))
        pygame.draw.rect(self.screen, self.colors['sidewalk'], (cx - sw_w//2, 0, sw_w, 600))
        if self.assets.images['road']:
            tex = pygame.transform.scale(self.assets.images['road'], (800, rw))
            tex_v = pygame.transform.scale(self.assets.images['road'], (rw, 600))
            self.screen.blit(tex, (0, cy - rw//2))
            self.screen.blit(tex_v, (cx - rw//2, 0))
            s = pygame.Surface((rw, rw))
            s.set_alpha(50); s.fill((0,0,0))
            self.screen.blit(s, (cx - rw//2, cy - rw//2))
        else:
            pygame.draw.rect(self.screen, (50, 50, 55), (0, cy - rw//2, 800, rw))
            pygame.draw.rect(self.screen, (50, 50, 55), (cx - rw//2, 0, rw, 600))
        pygame.draw.line(self.screen, self.colors['line_yellow'], (0, cy), (cx-rw//2, cy), 3)
        pygame.draw.line(self.screen, self.colors['line_yellow'], (cx+rw//2, cy), (800, cy), 3)
        pygame.draw.line(self.screen, self.colors['line_yellow'], (cx, 0), (cx, cy-rw//2), 3)
        pygame.draw.line(self.screen, self.colors['line_yellow'], (cx, cy+rw//2), (cx, 600), 3)
        off = self.stop_off
        pygame.draw.line(self.screen, (255,255,255), (cx-rw//2, cy-off), (cx, cy-off), 6) #N
        pygame.draw.line(self.screen, (255,255,255), (cx, cy+off), (cx+rw//2, cy+off), 6) #S
        pygame.draw.line(self.screen, (255,255,255), (cx-off, cy), (cx-off, cy+rw//2), 6) #W
        pygame.draw.line(self.screen, (255,255,255), (cx+off, cy-rw//2), (cx+off, cy), 6) #E
        for tx, ty in self.trees:
            self.draw_tree(tx, ty)

    def draw_sprite_car(self, car):
        offset_x = CAR_WIDTH // 2
        offset_y = CAR_LENGTH // 2
        if car.direction in ['left', 'right']:
            shadow_rect = pygame.Rect(car.x, car.y + CAR_WIDTH - 5, CAR_LENGTH, 10)
        else:
            shadow_rect = pygame.Rect(car.x, car.y + CAR_LENGTH - 5, CAR_WIDTH, 10)
        shadow_surf = pygame.Surface((shadow_rect.width + 10, shadow_rect.height + 10), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 80), (0, 0, shadow_rect.width, shadow_rect.height))
        self.screen.blit(shadow_surf, (shadow_rect.x, shadow_rect.y))
        sprite = self.assets.get_car_image(car.direction, car.sprite_index)
        if sprite:
            rect = sprite.get_rect(center=(car.x + offset_x, car.y + offset_y)) 
            self.screen.blit(sprite, rect)
            self.draw_car_lights(car, rect)
        else:
            self.draw_fallback_car(car)

    def draw_car_lights(self, car, rect):
        if car.is_braking:
            brake_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(brake_surf, (255, 0, 0, 150), (5, 5), 4) 
            if car.direction == 'up':
                self.screen.blit(brake_surf, (rect.left + 2, rect.bottom - 10))
                self.screen.blit(brake_surf, (rect.right - 12, rect.bottom - 10))
            elif car.direction == 'down':
                self.screen.blit(brake_surf, (rect.left + 2, rect.top))
                self.screen.blit(brake_surf, (rect.right - 12, rect.top))
            elif car.direction == 'right':
                self.screen.blit(brake_surf, (rect.left, rect.top + 2))
                self.screen.blit(brake_surf, (rect.left, rect.bottom - 12))
            elif car.direction == 'left':
                self.screen.blit(brake_surf, (rect.right - 10, rect.top + 2))
                self.screen.blit(brake_surf, (rect.right - 10, rect.bottom - 12))
        if not car.is_braking: 
            beam_len, beam_w = 60, 20
            beam_surf = pygame.Surface((beam_len, 40), pygame.SRCALPHA)
            pygame.draw.polygon(beam_surf, (255, 255, 200, 40), [(0, 10), (0, 30), (beam_len, 40), (beam_len, 0)])
            if car.direction == 'down':
                rotated_beam = pygame.transform.rotate(beam_surf, -90)
                self.screen.blit(rotated_beam, (rect.left - 5, rect.bottom))
                self.screen.blit(rotated_beam, (rect.right - 35, rect.bottom))
            elif car.direction == 'up':
                rotated_beam = pygame.transform.rotate(beam_surf, 90)
                self.screen.blit(rotated_beam, (rect.left - 5, rect.top - beam_len))
                self.screen.blit(rotated_beam, (rect.right - 35, rect.top - beam_len))
            elif car.direction == 'right':
                self.screen.blit(beam_surf, (rect.right, rect.top - 5))
                self.screen.blit(beam_surf, (rect.right, rect.bottom - 35))
            elif car.direction == 'left':
                rotated_beam = pygame.transform.rotate(beam_surf, 180)
                self.screen.blit(rotated_beam, (rect.left - beam_len, rect.top - 5))
                self.screen.blit(rotated_beam, (rect.left - beam_len, rect.bottom - 35))

    def draw_fallback_car(self, car):
        x, y, d, c = car.x, car.y, car.direction, car.color
        sc = tuple(max(0, val - 40) for val in c)
        if d in ['up', 'down']: w, h, depth = 28, 48, 10
        else: w, h, depth = 48, 28, 10
        self.draw_cube(x, y, w, h, depth, c, sc)

    def draw_3d_light(self, x, y, state):
        if self.assets.images['traffic_light']:
            sprite = self.assets.images['traffic_light']
            w, h = sprite.get_size()
            screen_x, screen_y = x - w // 2, y - h
            self.screen.blit(sprite, (screen_x, screen_y))
            box_top = screen_y + 10
            center_x = x
            spacing = 22 
            start_y = box_top + 15 
            if state == 'red': pygame.draw.circle(self.screen, (255,0,0), (center_x, start_y), 8)
            if state == 'green': pygame.draw.circle(self.screen, (0,255,0), (center_x, start_y + spacing*2), 8)
        else:
            box_x, box_y = x - 14, y - 100
            self.draw_cube(box_x, box_y, 28, 70, 8, (30,30,30), (10,10,10)) 
            r = (255, 50, 50) if state == 'red' else (70, 0, 0)
            g = (50, 255, 50) if state == 'green' else (0, 70, 0)
            
            #red
            pygame.draw.circle(self.screen, r, (box_x + 14, box_y + 12), 7)
            if r[0] > 100:
                 s = pygame.Surface((20, 20), pygame.SRCALPHA)
                 pygame.draw.circle(s, (*r, 50), (10, 10), 9)
                 self.screen.blit(s, (box_x + 4, box_y + 12 - 9))
            
            #green
            py = box_y + 12 + 40 
            pygame.draw.circle(self.screen, g, (box_x + 14, py), 7)
            if g[1] > 100:
                 s = pygame.Surface((20, 20), pygame.SRCALPHA)
                 pygame.draw.circle(s, (*g, 50), (10, 10), 9)
                 self.screen.blit(s, (box_x + 4, py - 9))

    def draw(self, active_index): 
        if not pygame.get_init(): return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                self.running = False
                pygame.quit()
                return False 
        self.draw_scenery()
        for p in self.particles:
            s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 100, 100, p.life), (p.size, p.size), p.size)
            self.screen.blit(s, (p.x, p.y))
        
        states = ['red'] * 4
        if 0 <= active_index <= 3:
            states[active_index] = 'green'
        
        off, cx, cy = self.stop_off, self.cx, self.cy
        
        self.draw_3d_light(cx - self.lane_w // 2 - 45, cy - off + 45, states[0]) 
        self.draw_3d_light(cx + self.lane_w // 2 + 47, cy + off + 80, states[1])
        self.draw_3d_light(cx + off  , cy - self.lane_w // 2 - 10, states[2]) 
        self.draw_3d_light(cx - off + 5, cy + self.lane_w // 2 + 130, states[3]) 
        
        q_counts = [len(l) for l in self.lanes]
        txt = self.queue_font.render(f"Q: {q_counts[0]}", True, (255, 255, 255))
        self.screen.blit(txt, (cx - self.lane_w - 70, cy - off - 10))
        txt = self.queue_font.render(f"Q: {q_counts[1]}", True, (255, 255, 255))
        self.screen.blit(txt, (cx + self.lane_w + 40, cy + off - 20))
        txt = self.queue_font.render(f"Q: {q_counts[2]}", True, (255, 255, 255))
        self.screen.blit(txt, (cx + off + 35, cy - self.lane_w - 25))
        txt = self.queue_font.render(f"Q: {q_counts[3]}", True, (255, 255, 255))
        self.screen.blit(txt, (cx - off - 50, cy + self.lane_w - 5))
        
        all_cars = []
        for q in self.lanes: all_cars.extend(q)
        all_cars.extend(self.leaving_cars)
        all_cars.sort(key=lambda c: c.y)
        for car in all_cars:
            self.draw_sprite_car(car)
        
        s = pygame.Surface((220, 90))
        s.set_alpha(200); s.fill((0,0,0))
        self.screen.blit(s, (10, 10))
        
        st_txt = "GREEN" 
        col = (0, 255, 0)
        if active_index == -1:
             st_txt = "ALL RED"
             col = (255, 50, 50)

        total_waiting = sum(len(l) for l in self.lanes)
        self.screen.blit(self.title_font.render(f"Waiting: {total_waiting}", True, (255,255,255)), (20, 20))
        self.screen.blit(self.font.render(st_txt, True, col), (20, 60))
        pygame.display.flip()
        return True

    def handle_events(self):
        if not pygame.get_init(): return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return False
        return True

    def close(self):
        if pygame.get_init():
            pygame.quit()