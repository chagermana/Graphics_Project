// =====================================================================
//  SUNRISE TO SUNSET  -  a family day: elephants, a car, a barbecue, and bedtime
//  Language : C++  (fixed-function OpenGL + GLUT / freeglut)
//
//  Graphics concepts used
//   1. Rendering pipeline  : model -> view (gluLookAt) -> projection
//                            (gluPerspective = viewing frustum) -> viewport
//   2. Transformations     : glTranslatef / glRotatef / glScalef
//                            (elephant legs/ears, person arms/legs, car wheels
//                             are all hierarchical rotations about a pivot)
//   3. Polygons            : walls, roof, ground are polygons (GL_QUADS)
//   4. Polygon filling     : scanline polygon fill (mountains) +
//                            Gouraud-shaded polygon (sky gradient)
//   5. Primitive algorithms (written by hand, plotted as pixels):
//        - Bresenham line      (sun rays, birds, HUD bar)
//        - DDA line            (sun rays, birds, HUD ticks)
//        - Midpoint circle     (sun, moon, clouds, HUD marker)
//   6. Shadows             : planar projected shadows (shadow matrix)
//                            + stencil buffer so overlaps are not doubled
//
//  THE STORY (timeOfDay goes 0..1 and loops; 0 = 04:48, noon = 0.30)
//   dawn       birds fly past, the car leaves for the shops
//   morning    an elephant family walks past, the kids cheer
//   midday     the car comes back with shopping, kids run to dad
//   afternoon  dad cooks on the barbecue (smoke)
//   sunset     the family eats outside, then goes in
//   night      moon + stars, lights go off one by one, "Z z z"
//
//  Build:
//    Linux/WSL: g++ sunrise_sunset.cpp -o sunrise -lglut -lGLU -lGL
//    Windows  : g++ sunrise_sunset.cpp -o sunrise -lfreeglut -lopengl32 -lglu32
//    macOS    : g++ sunrise_sunset.cpp -o sunrise -framework OpenGL -framework GLUT
//
//  Controls
//    SPACE pause/resume    + / -  faster / slower day
//    [ / ]  scrub time     LEFT/RIGHT/UP/DOWN  move camera
//    H  shadows on/off     R  reset     ESC  quit
// =====================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/glut.h>
  #include <GL/glu.h>
#endif

// ---------------------------------------------------------------------
//  Constants and global state
// ---------------------------------------------------------------------
static const float PI  = 3.14159265358979f;
static const float DEG = PI / 180.0f;          // degrees -> radians

static const float FOV         = 72.0f;        // vertical field of view
static const float HOUSE_ROT   = -20.0f;       // house turned a little (degrees)
static const float ROAD_ROT    = -20.0f;       // road slanted in the same direction as the driveway
static const float GROUND_HALF = 1000.0f;      // half size of the ground quad
static const float DAY_LEN     = 0.60f;        // sun is up for the first 60% of the cycle

static const float CAR_PARK_X  = -8.0f;        // where the car parks
static const float CAR_Z       = 9.3f;         // the road z-position
static const float ELE_Z       = -8.0f;        // the elephant trail (behind the house)

int   winW = 1100, winH = 700;

float timeOfDay   = 0.0f;                      // 0..1, then it loops
float daySpeed    = 1.0f / 70.0f;              // fraction of a day per second
bool  paused      = false;
bool  showShadows = true;
bool  shadowPass  = false;                     // true while drawing the shadow

float camYaw  = 0.0f;                          // degrees, left/right
float camDist = 24.0f;                         // distance from the house
const float LOOK_AT[3] = { 0.0f, 4.5f, 0.0f }; // camera looks at this point

int   lastTime = 0;
float rayAngle = 0.0f;                         // rotation of the sun rays
float windTime = 0.0f;                         // always-running clock for the grass sway

// Everything that changes with the time of day lives in this struct.
struct SunState {
    float dir[3];          // unit vector from the scene TOWARDS the sun
    float h;               // sin(elevation): 0 at the horizon, 1 straight up
    float dayK;            // 0 at horizon ... 1 high in the sky
    float vis;             // fades the sun light in/out around the horizon
    float night;           // 0 = day, 1 = full night
    float color[3];        // colour of the sunlight
    float intensity;       // strength of the sunlight
    float fillCol[3];      // colour of the soft fill light (sky / moon light)
    float ambient[3];      // ambient light
    float horizon[3];      // sky colour at the horizon
    float top[3];          // sky colour at the top of the screen
    float disc[3];         // colour of the sun disc
    float shadowAlpha;     // how dark the shadows are
    float glow;            // window glow 0..1 (low sun / night)
};
SunState g_sun;

// ---------------------------------------------------------------------
//  Small helpers
// ---------------------------------------------------------------------
float clamp01(float v)                { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
float lerpf(float a, float b, float t){ return a + (b - a) * t; }
void  mix3(float* out, const float* a, const float* b, float t)
{
    for (int i = 0; i < 3; i++) out[i] = lerpf(a[i], b[i], t);
}

// Set a colour (ignored during the shadow pass so shadows stay black)
void color3(float r, float g, float b)
{
    if (!shadowPass) glColor3f(r, g, b);
}

// Turn a point of the house (house coordinates) into world coordinates
void localToWorld(float lx, float lz, float& wx, float& wz)
{
    float a = HOUSE_ROT * DEG;
    wx =  lx * cosf(a) + lz * sinf(a);
    wz = -lx * sinf(a) + lz * cosf(a);
}

// ---------------------------------------------------------------------
//  1. SUN, MOON, SKY AND LIGHT COLOURS FOR THE CURRENT TIME
// ---------------------------------------------------------------------
void updateSun()
{
    SunState& s = g_sun;

    // sunT goes 0..1 while the sun is up, then stays at 1 (sun below the horizon)
    float sunT = timeOfDay / DAY_LEN;
    if (sunT > 1.0f) sunT = 1.0f;

    // The sun travels along an arc: azimuth from left (-38) to right (+38),
    // elevation from -8 degrees (below horizon) up to 32 and back down.
    float az = (-38.0f + 76.0f * sunT) * DEG;
    float el = (-8.0f + 40.0f * sinf(PI * sunT)) * DEG;

    s.dir[0] =  sinf(az) * cosf(el);
    s.dir[1] =  sinf(el);
    s.dir[2] = -cosf(az) * cosf(el);   // negative z = in front of the camera
    s.h      = s.dir[1];

    s.dayK = clamp01(s.h / 0.40f);               // 0 at horizon, 1 when high
    s.vis  = clamp01((s.h + 0.03f) / 0.12f);     // sunlight fades near horizon
    float twilight = clamp01((s.h + 0.14f) / 0.14f); // 0 = deep twilight, 1 = sun at horizon

    // night factor: full night after sunset until just before dawn
    if (timeOfDay < 0.04f)          s.night = 1.0f - timeOfDay / 0.04f;
    else if (timeOfDay > DAY_LEN)   s.night = clamp01((timeOfDay - DAY_LEN) / 0.06f);
    else                            s.night = 0.0f;

    // Sky colours: twilight -> dawn/dusk -> day -> (night)
    const float twiHz[3]  = { 0.30f, 0.22f, 0.38f };
    const float dawnHz[3] = { 1.00f, 0.58f, 0.32f };
    const float dayHz[3]  = { 0.75f, 0.88f, 1.00f };
    const float nitHz[3]  = { 0.05f, 0.07f, 0.17f };
    const float twiTop[3] = { 0.06f, 0.08f, 0.22f };
    const float dawnTop[3]= { 0.35f, 0.40f, 0.65f };
    const float dayTop[3] = { 0.22f, 0.50f, 0.92f };
    const float nitTop[3] = { 0.01f, 0.02f, 0.08f };
    const float twiAmb[3] = { 0.10f, 0.10f, 0.18f };
    const float dawnAmb[3]= { 0.28f, 0.22f, 0.26f };
    const float dayAmb[3] = { 0.30f, 0.33f, 0.40f };
    const float nitAmb[3] = { 0.07f, 0.08f, 0.15f };
    float tmp[3];

    mix3(tmp, twiHz,  dawnHz,  twilight); mix3(tmp, tmp, dayHz,  s.dayK); mix3(s.horizon, tmp, nitHz,  s.night);
    mix3(tmp, twiTop, dawnTop, twilight); mix3(tmp, tmp, dayTop, s.dayK); mix3(s.top,     tmp, nitTop, s.night);
    mix3(tmp, twiAmb, dawnAmb, twilight); mix3(tmp, tmp, dayAmb, s.dayK); mix3(s.ambient, tmp, nitAmb, s.night);

    // Sunlight: orange at the horizon, near white when high
    const float dawnSun[3] = { 1.00f, 0.55f, 0.25f };
    const float daySun[3]  = { 1.00f, 0.96f, 0.88f };
    mix3(s.color, dawnSun, daySun, s.dayK);
    s.intensity = s.vis * (0.5f + 0.5f * s.dayK);

    // Soft fill light from the camera side (sky light by day, blue moonlight at night)
    float fillScale = (0.15f + 0.85f * s.dayK) * (0.4f + 0.6f * twilight);
    const float dayFill[3] = { 0.35f * fillScale, 0.36f * fillScale, 0.40f * fillScale };
    const float nitFill[3] = { 0.10f, 0.14f, 0.28f };
    mix3(s.fillCol, dayFill, nitFill, s.night);

    // Sun disc colour
    const float discLow[3]  = { 1.00f, 0.40f, 0.12f };
    const float discHigh[3] = { 1.00f, 0.97f, 0.78f };
    mix3(s.disc, discLow, discHigh, s.dayK);

    // Shadows only when the sun is clearly above the horizon
    s.shadowAlpha = 0.5f * clamp01((s.h - 0.10f) / 0.12f);

    // Windows glow when the sun is low or it is dark
    s.glow = clamp01(1.0f - (s.h + 0.05f) / 0.35f);
}

// ---------------------------------------------------------------------
//  2. PIXEL ALGORITHMS (2D, in window pixel coordinates)
//     putPixel() must be called between glBegin(GL_POINTS) / glEnd()
//     hspan()    must be called between glBegin(GL_LINES)  / glEnd()
// ---------------------------------------------------------------------
void putPixel(int x, int y)
{
    glVertex2f(x + 0.5f, y + 0.5f);
}

// Horizontal run of pixels from x1 to x2 on row y
void hspan(int x1, int x2, int y)
{
    if (x2 < x1) return;
    glVertex2f((float)x1,        y + 0.5f);
    glVertex2f((float)(x2 + 1),  y + 0.5f);
}

// ---- Bresenham line algorithm (works in all octants) ----
void bresenhamLine(int x0, int y0, int x1, int y1)
{
    int dx =  abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;                       // error term

    while (true) {
        putPixel(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ---- DDA line algorithm ----
void ddaLine(float x0, float y0, float x1, float y1)
{
    float dx = x1 - x0;
    float dy = y1 - y0;
    int steps = (int)(fabsf(dx) > fabsf(dy) ? fabsf(dx) : fabsf(dy));
    if (steps == 0) { putPixel((int)lroundf(x0), (int)lroundf(y0)); return; }

    float xInc = dx / steps;
    float yInc = dy / steps;
    float x = x0, y = y0;
    for (int i = 0; i <= steps; i++) {
        putPixel((int)lroundf(x), (int)lroundf(y));
        x += xInc;
        y += yInc;
    }
}

// ---- Midpoint circle algorithm : outline (8-way symmetry) ----
void plot8(int cx, int cy, int x, int y)
{
    putPixel(cx + x, cy + y);  putPixel(cx - x, cy + y);
    putPixel(cx + x, cy - y);  putPixel(cx - x, cy - y);
    putPixel(cx + y, cy + x);  putPixel(cx - y, cy + x);
    putPixel(cx + y, cy - x);  putPixel(cx - y, cy - x);
}

void midpointCircle(int cx, int cy, int r)
{
    int x = 0, y = r;
    int d = 1 - r;                           // decision parameter
    while (x <= y) {
        plot8(cx, cy, x, y);
        x++;
        if (d < 0) {
            d += 2 * x + 1;                  // choose E
        } else {
            y--;
            d += 2 * (x - y) + 1;            // choose SE
        }
    }
}

// ---- Midpoint circle algorithm : filled disc (fills between symmetric points) ----
void midpointCircleFill(int cx, int cy, int r)
{
    int x = 0, y = r;
    int d = 1 - r;
    while (x <= y) {
        hspan(cx - x, cx + x, cy + y);
        hspan(cx - x, cx + x, cy - y);
        hspan(cx - y, cx + y, cy + x);
        hspan(cx - y, cx + y, cy - x);
        x++;
        if (d < 0) {
            d += 2 * x + 1;
        } else {
            y--;
            d += 2 * (x - y) + 1;
        }
    }
}

// ---- Scanline polygon fill ----
struct Pt { float x, y; };

void fillPolygon(const std::vector<Pt>& p)
{
    int n = (int)p.size();
    if (n < 3) return;

    float minY = p[0].y, maxY = p[0].y;
    for (int i = 1; i < n; i++) {
        if (p[i].y < minY) minY = p[i].y;
        if (p[i].y > maxY) maxY = p[i].y;
    }

    std::vector<float> xs;
    for (int y = (int)floorf(minY); y <= (int)ceilf(maxY); y++) {
        float yc = y + 0.5f;                 // sample at the pixel centre
        xs.clear();

        // find where this scanline crosses each edge
        for (int i = 0; i < n; i++) {
            Pt a = p[i], b = p[(i + 1) % n];
            if (a.y == b.y) continue;        // skip horizontal edges
            if (a.y > b.y) { Pt t = a; a = b; b = t; }
            if (yc >= a.y && yc < b.y)
                xs.push_back(a.x + (yc - a.y) * (b.x - a.x) / (b.y - a.y));
        }

        std::sort(xs.begin(), xs.end());
        // fill between pairs of crossings
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            int xa = (int)ceilf(xs[k]     - 0.5f);
            int xb = (int)floorf(xs[k + 1] - 0.5f);
            hspan(xa, xb, y);
        }
    }
}

// ---------------------------------------------------------------------
//  3. 2D MODE HELPERS
// ---------------------------------------------------------------------
void begin2D()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, (double)winW, 0.0, (double)winH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void end2D()
{
    glDisable(GL_BLEND);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
}

void drawText(float x, float y, const char* str)
{
    glRasterPos2f(x, y);
    for (const char* c = str; *c; ++c)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);
}

// ---------------------------------------------------------------------
//  4. THE STORY: who is where at which time
//     Every character follows a list of keyframes (time, x, z, facing).
//     Between two keyframes the position is interpolated (lerp).
// ---------------------------------------------------------------------
struct Key { float t, x, z, yaw; bool hidden; };   // hidden = indoors / away

struct PersonState { float x, z, yaw; bool hidden, moving; };

// The front door in world coordinates (people vanish into the doorway here)
const float DOOR_X = -1.06f, DOOR_Z = 2.91f;

// ---- DAD: out shopping, comes back by car, cooks, eats (sitting), carries the bags in ----
const Key dadKeys[] = {
    { 0.000f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 0.343f, -8.0f, 11.0f,  90.0f, false },   // gets out of the car (with shopping bags)
    { 0.362f, -5.2f, 11.0f,  90.0f, false },   // walks past the front of the car
    { 0.402f,  4.0f,  9.6f,  90.0f, false },   // round the barbecue
    { 0.420f,  4.0f,  8.0f, -90.0f, false },   // at the barbecue, puts the bags down
    { 0.520f,  4.0f,  8.0f, -90.0f, false },   // cooking
    { 0.565f,  9.3f,  5.7f, 180.0f, false },   // walks to the table
    { 0.571f,  9.3f,  4.95f,180.0f, false },   // sits on the bench
    { 0.625f,  9.3f,  4.95f,180.0f, false },   // eating
    { 0.631f,  9.3f,  5.7f, 180.0f, false },   // stands up
    { 0.665f,  4.2f,  7.4f, 180.0f, false },   // goes back for the bags
    { 0.672f,  4.2f,  7.4f, 180.0f, false },   // picks them up
    { 0.722f, -2.05f, 5.64f,180.0f, false },   // walks to the front of the house
    { 0.734f, DOOR_X, DOOR_Z,180.0f, false },  // into the doorway
    { 0.738f, DOOR_X, DOOR_Z,  0.0f, true  },  // inside
    { 1.000f, DOOR_X, DOOR_Z,  0.0f, true  }
};
// ---- MUM: looks after the kids, then sets the table ----
const Key mumKeys[] = {
    { 0.000f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 0.060f, DOOR_X, DOOR_Z,   0.0f, false },   // comes out
    { 0.068f, -1.57f,  4.32f,   0.0f, false },
    { 0.100f, -4.0f,   2.9f, -125.0f, false },
    { 0.330f, -4.0f,   2.9f, -125.0f, false },   // watching with the kids
    { 0.410f,  6.9f,   5.7f,  180.0f, false },   // to the table
    { 0.500f,  6.9f,   5.7f,  180.0f, false },   // waits, serving
    { 0.506f,  6.9f,   4.95f, 180.0f, false },   // sits
    { 0.625f,  6.9f,   4.95f, 180.0f, false },
    { 0.631f,  6.9f,   5.7f,  180.0f, false },   // stands up
    { 0.678f, -2.05f,  5.64f, 180.0f, false },   // walks to the front of the house
    { 0.690f, DOOR_X, DOOR_Z, 180.0f, false },
    { 0.694f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 1.000f, DOOR_X, DOOR_Z,   0.0f, true  }
};
// ---- KID 1 ----
const Key kid1Keys[] = {
    { 0.000f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 0.080f, DOOR_X, DOOR_Z,   0.0f, false },
    { 0.088f, -1.57f,  4.32f,   0.0f, false },
    { 0.113f, -7.5f,   3.5f, -110.0f, false },   // runs out to watch
    { 0.330f, -7.5f,   3.5f, -110.0f, false },
    { 0.350f, -4.0f,   8.0f,    0.0f, false },   // runs round the FRONT of the car
    { 0.363f, -4.0f,  11.6f,    0.0f, false },
    { 0.375f, -6.8f,  12.2f,    0.0f, false },   // greets dad
    { 0.420f, -6.8f,  12.2f,    0.0f, false },
    { 0.455f,  0.0f,  11.2f,    0.0f, false },   // goes to the table, round the barbecue
    { 0.485f,  6.2f,  10.6f,    0.0f, false },
    { 0.515f,  7.7f,   5.7f,  180.0f, false },
    { 0.521f,  7.7f,   4.95f, 180.0f, false },   // sits
    { 0.642f,  7.7f,   4.95f, 180.0f, false },
    { 0.648f,  7.7f,   5.7f,  180.0f, false },   // stands up
    { 0.690f, -2.05f,  5.64f, 180.0f, false },
    { 0.702f, DOOR_X, DOOR_Z, 180.0f, false },
    { 0.706f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 1.000f, DOOR_X, DOOR_Z,   0.0f, true  }
};
// ---- KID 2 ----
const Key kid2Keys[] = {
    { 0.000f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 0.085f, DOOR_X, DOOR_Z,   0.0f, false },
    { 0.093f, -1.57f,  4.32f,   0.0f, false },
    { 0.118f, -6.3f,   4.5f, -110.0f, false },
    { 0.330f, -6.3f,   4.5f, -110.0f, false },
    { 0.352f, -3.2f,   8.0f,    0.0f, false },   // runs round the FRONT of the car
    { 0.368f, -3.2f,  13.2f,    0.0f, false },
    { 0.382f, -9.0f,  13.0f,    0.0f, false },
    { 0.440f, -9.0f,  13.0f,    0.0f, false },
    { 0.470f, -0.5f,  12.2f,    0.0f, false },
    { 0.500f,  6.8f,  11.0f,    0.0f, false },
    { 0.525f,  8.5f,   5.7f,  180.0f, false },
    { 0.531f,  8.5f,   4.95f, 180.0f, false },   // sits
    { 0.652f,  8.5f,   4.95f, 180.0f, false },
    { 0.658f,  8.5f,   5.7f,  180.0f, false },
    { 0.702f, -2.05f,  5.64f, 180.0f, false },
    { 0.714f, DOOR_X, DOOR_Z, 180.0f, false },
    { 0.718f, DOOR_X, DOOR_Z,   0.0f, true  },
    { 1.000f, DOOR_X, DOOR_Z,   0.0f, true  }
};

PersonState evalPerson(const Key* k, int n)
{
    float t = timeOfDay;
    PersonState r;
    r.x = k[0].x; r.z = k[0].z; r.yaw = k[0].yaw; r.hidden = k[0].hidden; r.moving = false;
    if (t <= k[0].t) return r;

    for (int i = 0; i < n - 1; i++) {
        if (t < k[i + 1].t) {
            float u  = (t - k[i].t) / (k[i + 1].t - k[i].t);
            float dx = k[i + 1].x - k[i].x;
            float dz = k[i + 1].z - k[i].z;
            r.x = k[i].x + dx * u;
            r.z = k[i].z + dz * u;
            r.moving = (dx * dx + dz * dz > 0.01f);
            r.yaw    = r.moving ? atan2f(dx, dz) / DEG : k[i].yaw;
            r.hidden = k[i].hidden;
            return r;
        }
    }
    r.x = k[n - 1].x; r.z = k[n - 1].z; r.yaw = k[n - 1].yaw; r.hidden = k[n - 1].hidden;
    return r;
}

// Everything the story decides for the current time
struct Story {
    bool  carOn;   float carX;                  // the car
        bool  elephOn; float eleX;                  // cows visible / lead cow x
    float cowLie;                               // 0 = standing, 1 = lying asleep
    float cowZzz;                               // strength of the cows' "zzz"
                    // lead elephant x
    PersonState p[4];                           // 0 dad, 1 mum, 2 kid1, 3 kid2
    bool  grillOn;                              // barbecue is lit
    float lightA, lightB, lightBed;             // window lights (0 or 1, smooth)
    float zzz;                                  // strength of the "zzz"
    float doorTarget;                           // 0 = door closed, 1 = wide open
};
Story story;
float doorOpen = 0.0f;                              // eased towards story.doorTarget

float smoothStep01(float a, float b, float t)   // 0 before a, 1 after b
{
    return clamp01((t - a) / (b - a));
}

// 1 while t is between onT and offT (with a short fade)
float lightSwitch(float t, float onT, float offT)
{
    return smoothStep01(onT, onT + 0.01f, t) * (1.0f - smoothStep01(offT, offT + 0.01f, t));
}

bool carPosition(float t, float& x)
{
    if (t < 0.02f) { x = CAR_PARK_X; return true; }                  // parked overnight
    if (t < 0.055f) {                                                // drives away (speeding up)
        float u = (t - 0.02f) / 0.035f;
        x = lerpf(CAR_PARK_X, 48.0f, u * u);
        return true;
    }
    if (t < 0.30f) return false;                                     // away shopping
    if (t < 0.345f) {                                                // comes back (slowing down)
        float u = (t - 0.30f) / 0.045f;
        u = 1.0f - (1.0f - u) * (1.0f - u);
        x = lerpf(-30.0f, CAR_PARK_X, u);
        return true;
    }
    x = CAR_PARK_X;                                                  // parked
    return true;
}
// ---- the cows: 3 of them (mum, calf, dad) walk into the cowshed ----
const int   NUM_COWS      = 3;
const float COW_START[3]  = { -40.0f, -46.0f, -52.0f };  // where each cow enters (x)
const float COW_END[3]    = {  27.0f,  24.2f,  21.4f };  // where each cow stands in the shed
const float COW_Z[3]      = {  -8.0f,  -7.0f,  -8.9f };
const float COW_SCALE[3]  = {  0.95f,  0.65f,  0.90f };  // the middle one is a calf
const float COW_ARRIVE_T  = 0.29f;                       // time they reach the shed



// Day plan: asleep in the shed -> wake up -> walk out to the field (left)
//           -> away all day -> walk home -> turn to the camera -> sleep
const float COW_OUT_T  = 0.09f;   // first cow starts walking out (the others follow, +0.01 each)
const float COW_OUT_D  = 0.20f;   // how long the walk out takes
const float COW_BACK_T = 0.48f;   // first cow starts walking home (early evening)
const float COW_BACK_D = 0.14f;   // how long the walk home takes
float smoothStep01(float a, float b, float t);

float cowXAt(int i, float t)
{
    float outStart  = COW_OUT_T  + i * 0.01f;
    float backStart = COW_BACK_T + i * 0.01f;
    if (t < outStart)                                  // in the shed
        return COW_END[i];
    if (t < backStart)                                 // walking out, then away in the field
        return lerpf(COW_END[i], COW_START[i], clamp01((t - outStart) / COW_OUT_D));
    return lerpf(COW_START[i], COW_END[i],             // walking home, then in the shed
                 clamp01((t - backStart) / COW_BACK_D));
}

bool cowMoving(int i, float t)
{
    float outStart  = COW_OUT_T  + i * 0.01f;
    float backStart = COW_BACK_T + i * 0.01f;
    return (t > outStart  && t < outStart  + COW_OUT_D) ||
           (t > backStart && t < backStart + COW_BACK_D);
}

// 0 = facing +x, -90 = facing the camera, -180 = facing -x
float cowYaw(int i, float t)
{
    float outStart  = COW_OUT_T  + i * 0.01f;
    float backEnd   = COW_BACK_T + i * 0.01f + COW_BACK_D;
    if (t < 0.40f)                                     // morning: face camera, turn left, walk out
        return -90.0f - 90.0f * smoothStep01(outStart - 0.02f, outStart, t);
    return -90.0f * smoothStep01(backEnd, backEnd + 0.03f, t);   // evening: walk in, turn to camera
}

void updateStory()
{
    float t = timeOfDay;
    story.carOn = carPosition(t, story.carX);


    story.eleX    = cowXAt(0, t);

    // visible while in the shed or walking; hidden while out in the field
    story.elephOn = (t < 0.32f || t > 0.47f);
    // asleep at night, stand up at dawn; lie down again after coming home
    story.cowLie  = (t < 0.40f) ? 1.0f - smoothStep01(0.04f, 0.07f, t)
                                : smoothStep01(0.68f, 0.71f, t);
    // "zzz" while asleep (same value at t = 1 and t = 0, so the loop is seamless)
    story.cowZzz  = (t < 0.40f) ? 1.0f - smoothStep01(0.03f, 0.05f, t)
                                : smoothStep01(0.72f, 0.75f, t);
    
    story.p[0] = evalPerson(dadKeys,  (int)(sizeof(dadKeys)  / sizeof(Key)));
    story.p[1] = evalPerson(mumKeys,  (int)(sizeof(mumKeys)  / sizeof(Key)));
    story.p[2] = evalPerson(kid1Keys, (int)(sizeof(kid1Keys) / sizeof(Key)));
    story.p[3] = evalPerson(kid2Keys, (int)(sizeof(kid2Keys) / sizeof(Key)));

    // while the elephants pass, the kids turn to follow them
    for (int i = 2; i < 4; i++) {
        PersonState& p = story.p[i];
        if (!p.moving && t > 0.10f && t < 0.30f)
            p.yaw = atan2f(story.eleX - p.x, ELE_Z - p.z) / DEG;
    }

    story.grillOn = (t > 0.42f && t < 0.54f);

    story.lightA   = lightSwitch(t, 0.02f, 0.78f);
    story.lightB   = lightSwitch(t, 0.03f, 0.82f);
    story.lightBed = lightSwitch(t, 0.045f, 0.86f);

    // the door opens when somebody is close to it
    float target = 0.0f;
    for (int i = 0; i < 4; i++) {
        if (story.p[i].hidden) continue;
        float dx = story.p[i].x - DOOR_X, dz = story.p[i].z - DOOR_Z;
        float d  = sqrtf(dx * dx + dz * dz);
        if (d < 3.2f) target = std::max(target, clamp01(1.0f - (d - 0.8f) / 2.4f));
    }
    story.doorTarget = target;

    story.zzz = smoothStep01(0.87f, 0.89f, t) * (1.0f - smoothStep01(0.95f, 0.97f, t));
}

const char* eventText()
{
    float t = timeOfDay;
    if (t < 0.03f)  return "Before dawn - the family is asleep";
    if (t < 0.09f)  return "Sunrise - birds fly past, dad drives to the shops";
    if (t < 0.29f)  return "Morning - the cows head out to the field, the kids cheer";
    if (t < 0.345f) return "Midday - the car comes back with the shopping";
    if (t < 0.42f)  return "Dad is home - the kids run to greet him";
    if (t < 0.52f)  return "Afternoon - dad cooks on the barbecue";
    if (t < 0.625f) return "Sunset - dinner outside, the cows come home to the shed";
    if (t < 0.74f)  return "Everyone goes inside";
    if (t < 0.87f)  return "Night - the lights go off, one by one";
    if (t < 0.97f)  return "Goodnight...  zzz";
    return "Night";
}

// ---------------------------------------------------------------------
//  5. BACKGROUND (sky, stars, sun, moon, clouds, birds, mountains) - 2D
// ---------------------------------------------------------------------
struct Cloud { float x, yFrac, size, speed; };
Cloud clouds[5] = {
    { 150.0f, 0.30f, 34.0f,  9.0f },
    { 480.0f, 0.55f, 26.0f,  6.0f },
    { 800.0f, 0.40f, 40.0f, 11.0f },
    {1000.0f, 0.70f, 22.0f,  5.0f },
    { 300.0f, 0.78f, 28.0f,  7.0f }
};

struct Star { float fx, fy, b; };
const int NUM_STARS = 90;
Star stars[NUM_STARS];

void initStars()
{
    srand(12345);
    for (int i = 0; i < NUM_STARS; i++) {
        stars[i].fx = (float)rand() / (float)RAND_MAX;
        stars[i].fy = 0.25f + 0.75f * (float)rand() / (float)RAND_MAX;
        stars[i].b  = 0.5f  + 0.5f  * (float)rand() / (float)RAND_MAX;
    }
}

// Height of a mountain ridge (in pixels above the horizon) at panorama position u
float ridge(float u, int layer)
{
    if (layer == 0)
        return 52.0f + 28.0f * sinf(u * 0.0075f + 0.6f)
                     + 16.0f * sinf(u * 0.021f  + 2.0f)
                     +  8.0f * sinf(u * 0.050f  + 1.0f);
    return 36.0f + 22.0f * sinf(u * 0.0115f + 3.0f)
                 + 10.0f * sinf(u * 0.034f  + 0.5f)
                 +  5.0f * sinf(u * 0.070f);
}

void drawMountains(float hy, float shift)
{
    const SunState& s = g_sun;
    float light = lerpf(0.35f, 1.0f, s.vis * (0.4f + 0.6f * s.dayK)) * (1.0f - 0.65f * s.night);

    const float farBase[3]  = { 0.42f, 0.50f, 0.66f };
    const float nearBase[3] = { 0.24f, 0.33f, 0.38f };

    for (int layer = 0; layer < 2; layer++) {
        const float* base = (layer == 0) ? farBase : nearBase;
        float lit[3] = { base[0] * light, base[1] * light, base[2] * light };
        float col[3];
        mix3(col, lit, s.horizon, (layer == 0) ? 0.30f : 0.12f);

        // build the polygon: ridge line + two points below the horizon
        std::vector<Pt> poly;
        for (float x = -20.0f; x <= winW + 20.0f; x += 16.0f) {
            Pt p; p.x = x; p.y = hy + ridge(x - shift, layer);
            poly.push_back(p);
        }
        Pt a = { (float)winW + 20.0f, hy - 25.0f };
        Pt b = { -20.0f,              hy - 25.0f };
        poly.push_back(a);
        poly.push_back(b);

        glColor4f(col[0], col[1], col[2], 1.0f);
        glBegin(GL_LINES);
        fillPolygon(poly);
        glEnd();
    }
}

void drawClouds(float hy, float shift)
{
    const SunState& s = g_sun;
    const float dusk[3] = { 0.95f, 0.60f, 0.50f };
    const float day[3]  = { 1.00f, 1.00f, 1.00f };
    float c[3];
    mix3(c, dusk, day, s.dayK);
    float dim = (0.7f + 0.3f * s.vis) * (1.0f - 0.8f * s.night);

    // each cloud = 5 filled midpoint circles (dx, dy, radius in units of size)
    const float blob[5][3] = {
        { -1.1f, -0.25f, 0.70f }, { 0.0f, 0.0f, 1.00f }, { 1.1f, -0.20f, 0.80f },
        {  0.5f,  0.35f, 0.70f }, { -0.5f, 0.30f, 0.60f }
    };

    glColor4f(c[0] * dim, c[1] * dim, c[2] * dim, 1.0f);
    glBegin(GL_LINES);
    for (int i = 0; i < 5; i++) {
        float cx = clouds[i].x + shift * 0.6f;
        float cy = hy + clouds[i].yFrac * (winH - hy);
        float sz = clouds[i].size;
        for (int k = 0; k < 5; k++) {
            midpointCircleFill((int)lroundf(cx + blob[k][0] * sz),
                               (int)lroundf(cy + blob[k][1] * sz),
                               (int)lroundf(blob[k][2] * sz));
        }
    }
    glEnd();
}

void drawSun(float sx, float sy)
{
    const SunState& s = g_sun;
    int   R  = (int)(winH * 0.055f);          // radius of the sun in pixels
    int   cx = (int)lroundf(sx);
    int   cy = (int)lroundf(sy);

    // soft glow around the sun (additive blended triangle fan)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(s.disc[0], s.disc[1], s.disc[2], 0.55f);
    glVertex2f(sx, sy);
    for (int i = 0; i <= 48; i++) {
        float a = 2.0f * PI * i / 48.0f;
        glColor4f(s.disc[0], s.disc[1], s.disc[2], 0.0f);
        glVertex2f(sx + cosf(a) * 230.0f, sy + sinf(a) * 230.0f);
    }
    glEnd();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // rotating rays: even rays with DDA, odd rays with Bresenham
    glPointSize(2.0f);
    glColor4f(s.disc[0], s.disc[1], s.disc[2], 0.9f);
    glBegin(GL_POINTS);
    for (int i = 0; i < 16; i++) {
        float a  = rayAngle + i * (2.0f * PI / 16.0f);
        float r0 = (float)R + 10.0f;
        float r1 = (float)R + ((i % 2 == 0) ? 44.0f : 26.0f);
        float x0 = sx + cosf(a) * r0, y0 = sy + sinf(a) * r0;
        float x1 = sx + cosf(a) * r1, y1 = sy + sinf(a) * r1;
        if (i % 2 == 0)
            ddaLine(x0, y0, x1, y1);
        else
            bresenhamLine((int)lroundf(x0), (int)lroundf(y0),
                          (int)lroundf(x1), (int)lroundf(y1));
    }
    glEnd();
    glPointSize(1.0f);

    // thin ring (midpoint circle outline)
    glColor4f(s.disc[0], s.disc[1], s.disc[2], 0.7f);
    glBegin(GL_POINTS);
    midpointCircle(cx, cy, R + 5);
    glEnd();

    // the sun disc (midpoint circle, filled)
    glColor4f(s.disc[0], s.disc[1], s.disc[2], 1.0f);
    glBegin(GL_LINES);
    midpointCircleFill(cx, cy, R);
    glEnd();
}

// The moon: a pale midpoint-circle disc with a few darker craters
void drawMoon(float mx, float my)
{
    const SunState& s = g_sun;
    int R  = (int)(winH * 0.045f);
    int cx = (int)lroundf(mx);
    int cy = (int)lroundf(my);

    const float moonCol[3]   = { 0.93f, 0.94f, 0.85f };
    const float craterCol[3] = { 0.78f, 0.80f, 0.74f };
    float disc[3], crater[3];
    mix3(disc,   s.top, moonCol,   s.night);     // fades in with the night
    mix3(crater, s.top, craterCol, s.night);

    // soft halo
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(0.6f, 0.7f, 1.0f, 0.22f * s.night);
    glVertex2f(mx, my);
    for (int i = 0; i <= 48; i++) {
        float a = 2.0f * PI * i / 48.0f;
        glColor4f(0.6f, 0.7f, 1.0f, 0.0f);
        glVertex2f(mx + cosf(a) * 130.0f, my + sinf(a) * 130.0f);
    }
    glEnd();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(disc[0], disc[1], disc[2], 1.0f);
    glBegin(GL_LINES);
    midpointCircleFill(cx, cy, R);
    glEnd();

    glColor4f(crater[0], crater[1], crater[2], 1.0f);
    glBegin(GL_LINES);
    midpointCircleFill(cx - R / 3, cy + R / 4, R / 5);
    midpointCircleFill(cx + R / 3, cy - R / 5, R / 4);
    midpointCircleFill(cx + R / 6, cy + R / 2, R / 8);
    glEnd();
}

void drawStars(float hy, float shift, float alpha)
{
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_STARS; i++) {
        float x = fmodf(stars[i].fx * winW + shift + 10.0f * winW, (float)winW);
        float y = hy + stars[i].fy * (winH - hy);
        glColor4f(1.0f, 1.0f, 0.9f, alpha * stars[i].b);
        putPixel((int)x, (int)y);
    }
    glEnd();
    glPointSize(1.0f);
}

// A flock of birds: each bird is a "V" made of one DDA line and one Bresenham line
void drawBirds(float hy, float shift)
{
    float t = timeOfDay;
    if (t < 0.03f || t > 0.20f) return;
    float p = (t - 0.03f) / 0.17f;                       // 0..1 across the sky

    glPointSize(2.0f);
    glColor4f(0.08f, 0.08f, 0.10f, 0.9f);
    glBegin(GL_POINTS);
    for (int i = 0; i < 5; i++) {
        float x = -80.0f + p * (winW + 260.0f) - i * 38.0f + (i % 2) * 10.0f + shift * 0.5f;
        float y = hy + 0.62f * (winH - hy) + i * 14.0f - (i % 3) * 9.0f;
        float flap = 9.0f * sinf(timeOfDay * 1300.0f + i * 0.9f);   // wings up and down
        ddaLine(x - 13.0f, y + flap, x, y);
        bresenhamLine((int)lroundf(x), (int)lroundf(y),
                      (int)lroundf(x + 13.0f), (int)lroundf(y + flap));
    }
    glEnd();
    glPointSize(1.0f);
}

void drawBackground(float hy, bool sunVisible, float sx, float sy,
                    bool moonVisible, float mx, float my, float shift)
{
    const SunState& s = g_sun;
    begin2D();

    // sky gradient: a polygon with a colour on every corner (Gouraud shading)
    glBegin(GL_QUADS);
    glColor3fv(s.horizon); glVertex2f(0.0f, 0.0f);  glVertex2f((float)winW, 0.0f);
    glVertex2f((float)winW, hy);                    glVertex2f(0.0f, hy);
    glEnd();
    glBegin(GL_QUADS);
    glColor3fv(s.horizon); glVertex2f(0.0f, hy);    glVertex2f((float)winW, hy);
    glColor3fv(s.top);     glVertex2f((float)winW, (float)winH); glVertex2f(0.0f, (float)winH);
    glEnd();

    float starAlpha = clamp01(-s.h / 0.10f);
    if (starAlpha > 0.01f) drawStars(hy, shift, starAlpha);

    if (sunVisible && s.night < 0.5f) drawSun(sx, sy);
    if (moonVisible && s.night > 0.02f) drawMoon(mx, my);

    drawClouds(hy, shift);
    drawBirds(hy, shift);
    drawMountains(hy, shift);

    end2D();
}

// ---------------------------------------------------------------------
//  6. SHAPES FOR THE 3D SCENE  (all "blocky": boxes, cones, spheres)
// ---------------------------------------------------------------------
// A box centred at (cx,cy,cz) with sizes (sx,sy,sz)
void box(float cx, float cy, float cz, float sx, float sy, float sz)
{
    glPushMatrix();
    glTranslatef(cx, cy, cz);
    glScalef(sx, sy, sz);
    glutSolidCube(1.0);
    glPopMatrix();
}

// A cone pointing up, base at height y
void coneUp(float y, float radius, float height)
{
    glPushMatrix();
    glTranslatef(0.0f, y, 0.0f);
    glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);     // glutSolidCone points along +z
    glutSolidCone(radius, height, 16, 2);
    glPopMatrix();
}

void drawRoof()
{
    const float L = 4.4f, W = 3.5f;          // half length / half width
    const float y0 = 3.5f, y1 = 5.8f;        // eave height / ridge height
    const float dy = y1 - y0;

    color3(0.55f, 0.23f, 0.18f);
    glBegin(GL_QUADS);
    // front slope (+z)
    glNormal3f(0.0f, W, dy);
    glVertex3f(-L, y0,  W); glVertex3f( L, y0,  W);
    glVertex3f( L, y1, 0.0f); glVertex3f(-L, y1, 0.0f);
    // back slope (-z)
    glNormal3f(0.0f, W, -dy);
    glVertex3f(-L, y0, -W); glVertex3f(-L, y1, 0.0f);
    glVertex3f( L, y1, 0.0f); glVertex3f( L, y0, -W);
    glEnd();

    color3(0.42f, 0.18f, 0.14f);
    glBegin(GL_TRIANGLES);
    // right end (+x)
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(L, y0, W); glVertex3f(L, y0, -W); glVertex3f(L, y1, 0.0f);
    // left end (-x)
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-L, y0, -W); glVertex3f(-L, y0, W); glVertex3f(-L, y1, 0.0f);
    glEnd();

    // underside of the overhang
    color3(0.30f, 0.15f, 0.12f);
    glBegin(GL_QUADS);
    glNormal3f(0.0f, -1.0f, 0.0f);
    glVertex3f(-L, y0, -W); glVertex3f(L, y0, -W);
    glVertex3f( L, y0,  W); glVertex3f(-L, y0,  W);
    glEnd();
}

// Glass that glows (emission) when the light inside is on
void glassOn(float glow)
{
    const float day[3]  = { 0.55f, 0.72f, 0.90f };
    const float warm[3] = { 1.00f, 0.85f, 0.45f };
    float c[3];
    mix3(c, day, warm, glow);
    glColor3fv(c);
    float e[4] = { 0.90f * glow, 0.72f * glow, 0.32f * glow, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, e);
}

void glassOff()
{
    float e[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, e);
}

void windowFront(float x, float y, float lit)   // window on the front wall (z = 3)
{
    color3(0.30f, 0.20f, 0.12f);
    box(x, y, 3.02f, 1.7f, 1.5f, 0.06f);
    glassOn(g_sun.glow * lit);
    box(x, y, 3.04f, 1.4f, 1.2f, 0.08f);
    glassOff();
    color3(0.30f, 0.20f, 0.12f);
    box(x, y, 3.09f, 1.4f, 0.07f, 0.04f);
    box(x, y, 3.09f, 0.07f, 1.2f, 0.04f);
}

void windowSide(float z, float y, float lit)    // window on the right wall (x = 4)
{
    color3(0.30f, 0.20f, 0.12f);
    box(4.02f, y, z, 0.06f, 1.5f, 1.7f);
    glassOn(g_sun.glow * lit);
    box(4.04f, y, z, 0.08f, 1.2f, 1.4f);
    glassOff();
    color3(0.30f, 0.20f, 0.12f);
    box(4.09f, y, z, 0.04f, 0.07f, 1.4f);
    box(4.09f, y, z, 0.04f, 1.2f, 0.07f);
}

void drawHouse()
{
    glPushMatrix();
    glRotatef(HOUSE_ROT, 0.0f, 1.0f, 0.0f);  // transformation: rotate the house

    // walls
    color3(0.93f, 0.85f, 0.70f);
    box(0.0f, 1.75f, 0.0f, 8.0f, 3.5f, 6.0f);

    drawRoof();

    // chimney
    color3(0.70f, 0.38f, 0.28f);
    box(2.4f, 5.3f, -0.8f, 0.9f, 2.2f, 0.9f);
    color3(0.30f, 0.30f, 0.32f);
    box(2.4f, 6.45f, -0.8f, 1.1f, 0.14f, 1.1f);

    if (!shadowPass) {
        // door step, door and knob
        color3(0.60f, 0.60f, 0.62f);
        box(0.0f, 0.08f, 3.5f, 2.0f, 0.16f, 1.0f);
        // dark doorway, and the door leaf which swings open on its hinge
        color3(0.10f, 0.07f, 0.05f);
        box(0.0f, 1.1f, 3.02f, 1.3f, 2.2f, 0.04f);
        glPushMatrix();
        glTranslatef(-0.65f, 0.0f, 3.04f);              // hinge on the left edge
        glRotatef(-85.0f * doorOpen, 0.0f, 1.0f, 0.0f); // swings outwards
        color3(0.40f, 0.22f, 0.10f);
        box(0.65f, 1.1f, 0.0f, 1.3f, 2.2f, 0.08f);
        color3(0.85f, 0.70f, 0.20f);
        glPushMatrix();
        glTranslatef(1.10f, 1.05f, 0.07f);
        glutSolidSphere(0.07, 10, 10);
        glPopMatrix();
        glPopMatrix();

        // windows (each has its own light: left = living room, right = kitchen, side = bedroom)
        windowFront(-2.5f, 2.1f, story.lightA);
        windowFront( 2.5f, 2.1f, story.lightB);
        windowSide(0.0f, 2.1f, story.lightBed);
    }
    glPopMatrix();
}

// An acacia: thin trunk that forks, flat umbrella-shaped canopy
void flatBlob(float cx, float cy, float cz, float rx, float ry, float rz)
{
    glPushMatrix();
    glTranslatef(cx, cy, cz);
    glScalef(rx, ry, rz);
    glutSolidSphere(1.0, 16, 10);
    glPopMatrix();
}

void drawTree(float x, float z, float s)
{
    glPushMatrix();
    glTranslatef(x, 0.0f, z);
    glScalef(s, s, s);                       // transformation: scale the tree

    // trunk, then two branches that lean outwards (rotate about the fork)
    color3(0.38f, 0.28f, 0.16f);
    box(0.0f, 1.5f, 0.0f, 0.5f, 3.0f, 0.5f);
    glPushMatrix();
    glTranslatef(0.0f, 2.9f, 0.0f);
    glRotatef(30.0f, 0.0f, 0.0f, 1.0f);
    box(0.0f, 1.0f, 0.0f, 0.3f, 2.0f, 0.3f);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0.0f, 2.9f, 0.0f);
    glRotatef(-35.0f, 0.0f, 0.0f, 1.0f);
    box(0.0f, 1.0f, 0.0f, 0.3f, 2.0f, 0.3f);
    glPopMatrix();

    // flat canopy: squashed spheres at the ends of the branches
    color3(0.40f, 0.55f, 0.18f);
    flatBlob(-1.0f, 4.80f,  0.0f, 2.6f, 0.55f, 2.6f);
    flatBlob( 1.15f, 4.70f, 0.3f, 2.8f, 0.60f, 2.8f);
    color3(0.45f, 0.60f, 0.20f);
    flatBlob( 0.1f, 5.25f, -0.2f, 2.2f, 0.45f, 2.2f);
    glPopMatrix();
}

void drawBush(float x, float z)
{
    glPushMatrix();
    glTranslatef(x, 0.7f, z);
    glScalef(1.3f, 0.9f, 1.3f);
    color3(0.15f, 0.42f, 0.17f);
    glutSolidSphere(0.9, 16, 12);
    glPopMatrix();
}

// Tree positions (shared by drawing and by the grass shadow test)
struct TreeDef { float x, z, s; };
const int      NUM_TREES = 4;
const TreeDef  TREES[NUM_TREES] = {
    { -10.0f,   3.0f, 1.00f },
    {  11.0f,  -4.0f, 1.25f },
    { -17.0f, -10.0f, 1.50f },
    {  19.0f, -12.0f, 1.40f }
};

// =====================================================================
//  GRASS: real blades (3 triangles each: base quad + tapered tip),
//  clustered in tufts, with a gentle wind sway. The blades are stored in
//  vertex arrays that are rebuilt every frame. The SAME array is drawn
//  twice: once lit, and once flattened by the shadow matrix, so every
//  blade casts its own thin shadow onto the ground.
// =====================================================================
struct Tuft  { float x, z, dark; };                    // dark = current shadow strength 0..alpha
struct Blade {
    float x, z;            // root position
    float h, hw;           // height, half width at the root
    float phi;             // direction of the blade's width
    float psi, lean;       // direction and amount of lean
    float phase;           // wind phase
    float col[3];          // base colour of this blade
    int   tuft;            // which tuft it belongs to
};
std::vector<Tuft>  g_tufts;
std::vector<Blade> g_blades;
std::vector<float> g_grassPos, g_grassCol;

float frand() { return (float)rand() / (float)RAND_MAX; }

// Grass is only planted where nothing is built: not under the house, on the
// path to the door or on the road (all in the house's rotated frame).
bool grassAllowed(float x, float z)
{
    if (x > 18.5f && x < 33.0f && z > -12.0f && z < -4.0f) return false;   // cowshed

    float a  = HOUSE_ROT * DEG;
    float lx = x * cosf(a) - z * sinf(a);
    float lz = x * sinf(a) + z * cosf(a);
    if (fabsf(lx) < 5.0f && fabsf(lz) < 4.2f)                  return false;   // house
    if (fabsf(lx) < 1.6f && lz > 3.0f && lz < CAR_Z - 1.2f)    return false;   // path
    if (fabsf(lz - CAR_Z) < 2.1f && fabsf(lx) < 62.0f)         return false;   // road
    return true;
}

void addTuft(float x, float z, int count, float scale)
{
    if (!grassAllowed(x, z)) return;
    Tuft t; t.x = x; t.z = z; t.dark = 0.0f;
    g_tufts.push_back(t);
    int ti = (int)g_tufts.size() - 1;

    // a few greens so the lawn is not one flat colour
    static const float pal[4][3] = {
        { 0.27f, 0.65f, 0.22f },   // same as the ground
        { 0.20f, 0.54f, 0.18f },   // darker
        { 0.40f, 0.74f, 0.26f },   // lighter
        { 0.62f, 0.72f, 0.24f }    // a few dry yellowish blades
    };

    for (int i = 0; i < count; i++) {
        Blade b;
        b.x = x + (frand() - 0.5f) * 0.7f;
        b.z = z + (frand() - 0.5f) * 0.7f;
        b.h   = (0.28f + 0.30f * frand()) * scale;
        b.hw  = (0.055f + 0.030f * frand()) * scale;
        b.phi = frand() * PI;
        b.psi = atan2f(b.z - z, b.x - x) + (frand() - 0.5f);      // fan out from the tuft centre
        b.lean  = 0.20f + 0.45f * frand();
        b.phase = frand() * 6.2832f;
        float r = frand();
        const float* c = (r < 0.55f) ? pal[0] : (r < 0.80f) ? pal[1] : (r < 0.95f) ? pal[2] : pal[3];
        for (int k = 0; k < 3; k++) b.col[k] = c[k];
        b.tuft = ti;
        g_blades.push_back(b);
    }
}

void initGrass()
{
    srand(777);
    g_tufts.clear();
    g_blades.clear();

    // dense, detailed lawn near the house (jittered grid)
    for (float x = -30.0f; x < 30.0f; x += 1.5f)
        for (float z = -22.0f; z < 32.0f; z += 1.5f)
            addTuft(x + (frand() - 0.5f) * 1.4f, z + (frand() - 0.5f) * 1.4f, 4, 1.0f);

    // sparser, slightly bigger tufts further out so the lawn fades gently into the ground
    for (float x = -60.0f; x < 60.0f; x += 3.0f)
        for (float z = -45.0f; z < 50.0f; z += 3.0f) {
            if (x > -30.0f && x < 30.0f && z > -22.0f && z < 32.0f) continue;   // inner lawn already planted
            float ex = fabsf(x) / 60.0f, ez = fabsf(z - 2.0f) / 48.0f;
            float edge = (ex > ez ? ex : ez);                                  // 0 centre ... 1 edge
            if (frand() < edge * edge) continue;                               // thin out toward the edge
            addTuft(x + (frand() - 0.5f) * 2.8f, z + (frand() - 0.5f) * 2.8f, 3, 1.35f);
        }
}

// ---- Does a ray from p towards the sun hit a convex shape made of planes n.p <= c ? ----
bool rayHitConvex(const float p[3], const float d[3], const float pl[][4], int n)
{
    float t0 = 0.01f, t1 = 1.0e9f;
    for (int i = 0; i < n; i++) {
        float denom = pl[i][0] * d[0] + pl[i][1] * d[1] + pl[i][2] * d[2];
        float num   = pl[i][3] - (pl[i][0] * p[0] + pl[i][1] * p[1] + pl[i][2] * p[2]);
        if (fabsf(denom) < 1.0e-6f) { if (num < 0.0f) return false; continue; }
        float t = num / denom;
        if (denom > 0.0f) { if (t < t1) t1 = t; }
        else              { if (t > t0) t0 = t; }
        if (t0 > t1) return false;
    }
    return t1 > t0;
}

// Ray against an ellipsoid (centre c, radii r): used for the flat tree canopies
bool rayHitEllipsoid(const float p[3], const float d[3], const float c[3], const float r[3])
{
    float q[3], e[3];
    for (int i = 0; i < 3; i++) { q[i] = (p[i] - c[i]) / r[i]; e[i] = d[i] / r[i]; }
    float a  = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    float b  = q[0] * e[0] + q[1] * e[1] + q[2] * e[2];
    float cc = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0f;
    float disc = b * b - a * cc;
    if (disc <= 0.0f) return false;
    return (-b + sqrtf(disc)) / a > 0.0f;
}

// Is the grass at (x,z) in the shadow of the house or a tree?
bool grassInShadow(float x, float z)
{
    const SunState& s = g_sun;
    float p[3] = { x, 0.25f, z };
    float d[3] = { s.dir[0], s.dir[1], s.dir[2] };

    // house: move the ray into the house's own (rotated) frame, then test box + roof
    float a = HOUSE_ROT * DEG, ca = cosf(a), sa = sinf(a);
    float lp[3] = { p[0] * ca - p[2] * sa, p[1], p[0] * sa + p[2] * ca };
    float ld[3] = { d[0] * ca - d[2] * sa, d[1], d[0] * sa + d[2] * ca };
    static const float body[6][4] = {
        { 1, 0, 0, 4.0f }, { -1, 0, 0, 4.0f }, { 0, 1, 0, 3.5f },
        { 0, -1, 0, 0.0f }, { 0, 0, 1, 3.0f }, { 0, 0, -1, 3.0f } };
    static const float roof[5][4] = {
        { 1, 0, 0, 4.4f }, { -1, 0, 0, 4.4f }, { 0, -1, 0, -3.5f },
        { 0, 3.5f, 2.3f, 20.3f }, { 0, 3.5f, -2.3f, 20.3f } };
    if (rayHitConvex(lp, ld, body, 6)) return true;
    if (rayHitConvex(lp, ld, roof, 5))  return true;

        // cowshed roof (world coordinates, so use p and d)
    static const float shedRoof[6][4] = {
        { 1, 0, 0, 31.5f }, { -1, 0, 0, -19.5f }, { 0, 1, 0, 5.0f },
        { 0, -1, 0, -3.4f }, { 0, 0, 1, -4.9f }, { 0, 0, -1, 11.1f } };
    if (rayHitConvex(p, d, shedRoof, 6)) return true;

    // trees: the three flat canopy blobs of drawTree()
    static const float blob[3][6] = {
        { -1.0f,  4.80f,  0.0f, 2.6f, 0.55f, 2.6f },
        {  1.15f, 4.70f,  0.3f, 2.8f, 0.60f, 2.8f },
        {  0.1f,  5.25f, -0.2f, 2.2f, 0.45f, 2.2f } };
    for (int i = 0; i < NUM_TREES; i++)
        for (int k = 0; k < 3; k++) {
            float sc = TREES[i].s;
            float c[3] = { TREES[i].x + blob[k][0] * sc, blob[k][1] * sc, TREES[i].z + blob[k][2] * sc };
            float r[3] = { blob[k][3] * sc, blob[k][4] * sc, blob[k][5] * sc };
            if (rayHitEllipsoid(p, d, c, r)) return true;
        }
    return false;
}

// Tufts inside a shadow get darker (smoothly, so they do not pop when the sun moves)
void updateGrassShadows()
{
    const SunState& s = g_sun;
    bool on = showShadows && s.shadowAlpha > 0.01f;
    for (size_t i = 0; i < g_tufts.size(); i++) {
        Tuft& t = g_tufts[i];
        float target = (on && grassInShadow(t.x, t.z)) ? s.shadowAlpha : 0.0f;
        t.dark += (target - t.dark) * 0.35f;
    }
}

static void pushVert(float x, float y, float z, const float* c, float f)
{
    g_grassPos.push_back(x); g_grassPos.push_back(y); g_grassPos.push_back(z);
    g_grassCol.push_back(clamp01(c[0] * f));
    g_grassCol.push_back(clamp01(c[1] * f));
    g_grassCol.push_back(clamp01(c[2] * f));
}

// Build the blade triangles for this frame (with wind sway and shadow darkening)
void buildGrass()
{
    g_grassPos.clear();
    g_grassCol.clear();
    g_grassPos.reserve(g_blades.size() * 27);
    g_grassCol.reserve(g_blades.size() * 27);

    for (size_t i = 0; i < g_blades.size(); i++) {
        const Blade& b = g_blades[i];
        float lit = 1.0f - g_tufts[b.tuft].dark;           // same darkness as the ground shadow

        // wind: slow wave across the lawn + small flutter. The wind blows mostly towards +x
        float sw = sinf(windTime * 1.7f + b.x * 0.32f + b.z * 0.21f + b.phase * 0.3f) * 0.20f * b.h
                 + sinf(windTime * 4.1f + b.phase) * 0.05f * b.h;
        float swx = 0.9f * sw, swz = 0.4f * sw;

        float lx = cosf(b.psi) * b.lean * b.h;             // the blade's natural lean
        float lz = sinf(b.psi) * b.lean * b.h;
        float wx = cosf(b.phi), wz = sinf(b.phi);          // width direction

        float midY = 0.55f * b.h;
        float mx = b.x + lx * 0.30f + swx * 0.30f;
        float mz = b.z + lz * 0.30f + swz * 0.30f;
        float tx = b.x + lx + swx;
        float tz = b.z + lz + swz;
        float w0 = b.hw, w1 = b.hw * 0.68f;

        // dark roots, lighter tips: gives depth without looking realistic
        float cb[3], cm[3], ct[3];
        for (int k = 0; k < 3; k++) { cb[k] = b.col[k] * 0.60f; cm[k] = b.col[k] * 0.95f; ct[k] = b.col[k] * 1.12f; }

        // base quad = 2 triangles, tapered tip = 1 triangle
        pushVert(b.x - wx * w0, 0.0f, b.z - wz * w0, cb, lit);
        pushVert(b.x + wx * w0, 0.0f, b.z + wz * w0, cb, lit);
        pushVert(mx + wx * w1, midY, mz + wz * w1, cm, lit);

        pushVert(b.x - wx * w0, 0.0f, b.z - wz * w0, cb, lit);
        pushVert(mx + wx * w1, midY, mz + wz * w1, cm, lit);
        pushVert(mx - wx * w1, midY, mz - wz * w1, cm, lit);

        pushVert(mx - wx * w1, midY, mz - wz * w1, cm, lit);
        pushVert(mx + wx * w1, midY, mz + wz * w1, cm, lit);
        pushVert(tx, b.h, tz, ct, lit);
    }
}

// Draw the blade array. withColor = false is used for the shadow pass.
void drawGrassArrays(bool withColor)
{
    if (g_grassPos.empty()) return;
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &g_grassPos[0]);
    if (withColor) {
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_FLOAT, 0, &g_grassCol[0]);
    }
    glNormal3f(0.0f, 1.0f, 0.0f);          // blades are lit like the ground, from every side
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(g_grassPos.size() / 3));
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

// ---- Garden props: picnic table and barbecue ----
void drawProps()
{
    // picnic table at (8, 3.8) with two benches
    color3(0.55f, 0.38f, 0.20f);
    box(8.0f, 1.00f, 3.8f, 2.8f, 0.12f, 1.3f);        // table top
    box(8.0f, 0.55f, 2.7f, 2.8f, 0.10f, 0.4f);        // bench (far)
    box(8.0f, 0.55f, 4.9f, 2.8f, 0.10f, 0.4f);        // bench (near)
    box(6.9f, 0.50f, 3.8f, 0.12f, 1.0f, 1.1f);        // table legs
    box(9.1f, 0.50f, 3.8f, 0.12f, 1.0f, 1.1f);
    color3(0.95f, 0.95f, 0.95f);
    box(7.2f, 1.08f, 3.8f, 0.35f, 0.04f, 0.35f);      // plates
    box(7.9f, 1.08f, 3.8f, 0.35f, 0.04f, 0.35f);
    box(8.6f, 1.08f, 3.8f, 0.35f, 0.04f, 0.35f);
    box(9.3f, 1.08f, 3.8f, 0.35f, 0.04f, 0.35f);

    // the shopping bags dad put down next to the barbecue (until he picks them up again)
    if (timeOfDay >= 0.42f && timeOfDay < 0.672f) {
        color3(0.85f, 0.65f, 0.25f);
        box(4.0f, 0.20f, 6.6f, 0.30f, 0.40f, 0.22f);
        color3(0.30f, 0.60f, 0.85f);
        box(4.4f, 0.20f, 6.4f, 0.30f, 0.40f, 0.22f);
    }

    // barbecue at (2.6, 8)
    color3(0.15f, 0.15f, 0.17f);
    box(2.25f, 0.4f, 7.75f, 0.08f, 0.8f, 0.08f);      // legs
    box(2.95f, 0.4f, 7.75f, 0.08f, 0.8f, 0.08f);
    box(2.25f, 0.4f, 8.25f, 0.08f, 0.8f, 0.08f);
    box(2.95f, 0.4f, 8.25f, 0.08f, 0.8f, 0.08f);
    box(2.6f, 0.95f, 8.0f, 1.0f, 0.35f, 0.6f);        // body
    if (story.grillOn && !shadowPass) {
        color3(1.0f, 0.4f, 0.1f);                       // glowing coals
        float e[4] = { 0.8f, 0.25f, 0.05f, 1.0f };
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, e);
        box(2.6f, 1.15f, 8.0f, 0.9f, 0.06f, 0.5f);
        glassOff();
    } else {
        color3(0.25f, 0.10f, 0.05f);
        box(2.6f, 1.15f, 8.0f, 0.9f, 0.06f, 0.5f);
    }
}

// ---- A blocky cow. It faces +x. lie = 0 standing ... 1 lying down asleep ----
// ---- A blocky cow. It faces +x. lie = 0 standing ... 1 lying down asleep ----
void drawCow(float x, float z, float scale, float walkPhase,
             float swingAmt, float lie, int variant, float yaw)
{
    const float white[3]  = { 0.95f, 0.94f, 0.90f };
    const float blackP[3] = { 0.10f, 0.10f, 0.11f };
    const float brownP[3] = { 0.45f, 0.25f, 0.12f };
    const float* patch = (variant == 1) ? brownP : blackP;
    const float patchX[3][2] = { { -0.45f, 0.65f }, { 0.10f, -0.70f }, { -0.10f, 0.75f } };
    int   v = variant % 3;

    float swing     = 20.0f * sinf(walkPhase) * swingAmt;
    float tailSwish = 15.0f * sinf(timeOfDay * 600.0f + variant * 2.0f);
    float bodyY = lerpf(1.35f, 0.55f, lie);       // body sinks when lying down
    float hipY  = bodyY - 0.45f;
    float headY = lerpf(1.60f, 0.70f, lie);       // head rests low when asleep

    glPushMatrix();
    glTranslatef(x, 0.0f, z);
    glRotatef(yaw, 0.0f, 1.0f, 0.0f);        // 0 = facing +x, -90 = facing the camera
    glScalef(scale, scale, scale);

    // legs: pivot at the hip, diagonal pairs swing together (same idea as before)
    if (lie < 0.5f) {
        for (int i = 0; i < 4; i++) {
            float lx = (i < 2) ? 0.75f : -0.75f;
            float lz = (i % 2 == 0) ? 0.3f : -0.3f;
            float a  = ((i == 0 || i == 3) ? swing : -swing);
            glPushMatrix();
            glTranslatef(lx, hipY, lz);
            glRotatef(a, 0.0f, 0.0f, 1.0f);
            color3(white[0], white[1], white[2]);
            box(0.0f, -hipY * 0.5f, 0.0f, 0.22f, hipY, 0.22f);
            color3(0.15f, 0.12f, 0.10f);                      // hoof
            box(0.0f, -hipY + 0.06f, 0.0f, 0.25f, 0.12f, 0.25f);
            glPopMatrix();
        }
    } else {
        // lying: legs folded under the body
        color3(white[0], white[1], white[2]);
        box( 0.9f, 0.12f,  0.40f, 0.55f, 0.20f, 0.20f);
        box( 0.9f, 0.12f, -0.40f, 0.55f, 0.20f, 0.20f);
        box(-0.7f, 0.12f,  0.45f, 0.50f, 0.22f, 0.22f);
        box(-0.7f, 0.12f, -0.45f, 0.50f, 0.22f, 0.22f);
    }

    // body + two patches
    color3(white[0], white[1], white[2]);
    box(0.0f, bodyY, 0.0f, 2.2f, 1.0f, 0.9f);
    color3(patch[0], patch[1], patch[2]);
    box(patchX[v][0], bodyY + 0.12f, 0.0f, 0.6f, 0.8f, 0.94f);
    box(patchX[v][1], bodyY + 0.20f, 0.0f, 0.4f, 0.7f, 0.94f);

    // udder
    color3(0.95f, 0.70f, 0.70f);
    box(-0.55f, bodyY - 0.50f, 0.0f, 0.4f, 0.2f, 0.35f);

    // neck, head, snout, horns, ears, eyes
    float neckTop = bodyY + 0.25f;
    color3(white[0], white[1], white[2]);
    box(1.25f, (neckTop + headY) * 0.5f, 0.0f, 0.55f, fabsf(neckTop - headY) + 0.5f, 0.55f);
    box(1.55f, headY, 0.0f, 0.60f, 0.55f, 0.55f);
    color3(0.95f, 0.70f, 0.70f);
    box(1.90f, headY - 0.10f, 0.0f, 0.25f, 0.30f, 0.45f);
    color3(0.90f, 0.85f, 0.70f);
    box(1.45f, headY + 0.35f,  0.20f, 0.08f, 0.18f, 0.08f);
    box(1.45f, headY + 0.35f, -0.20f, 0.08f, 0.18f, 0.08f);
    color3(patch[0], patch[1], patch[2]);
    box(1.40f, headY + 0.20f,  0.38f, 0.20f, 0.10f, 0.25f);
    box(1.40f, headY + 0.20f, -0.38f, 0.20f, 0.10f, 0.25f);
    if (!shadowPass) {
        color3(0.05f, 0.05f, 0.05f);
        if (lie < 0.7f) {                                      // eyes open
            box(1.72f, headY + 0.12f,  0.28f, 0.06f, 0.08f, 0.04f);
            box(1.72f, headY + 0.12f, -0.28f, 0.06f, 0.08f, 0.04f);
        } else {                                               // eyes closed (thin lines)
            box(1.72f, headY + 0.12f,  0.28f, 0.08f, 0.015f, 0.04f);
            box(1.72f, headY + 0.12f, -0.28f, 0.08f, 0.015f, 0.04f);
        }
    }

    // tail swishes about its pivot
    glPushMatrix();
    glTranslatef(-1.1f, bodyY + 0.3f, 0.0f);
    glRotatef(tailSwish, 1.0f, 0.0f, 0.0f);
    color3(white[0], white[1], white[2]);
    box(0.0f, -0.45f, 0.0f, 0.10f, 0.90f, 0.10f);
    color3(patch[0], patch[1], patch[2]);
    box(0.0f, -0.95f, 0.0f, 0.14f, 0.20f, 0.14f);
    glPopMatrix();

    glPopMatrix();
}

// ---- Cowshed: open front and open left side (so cows can walk in), seen from the camera ----
void drawCowshed()
{
    const float cx = 25.5f, cz = -8.0f;

    // straw floor
    color3(0.80f, 0.70f, 0.40f);
    box(cx, 0.03f, cz, 11.0f, 0.06f, 5.4f);

    // back wall + right wall
    color3(0.62f, 0.22f, 0.16f);
    box(cx, 1.75f, -10.5f, 11.0f, 3.5f, 0.25f);
    box(30.9f, 1.75f, cz, 0.25f, 3.5f, 5.0f);

    // posts along the front
    color3(0.40f, 0.28f, 0.16f);
    box(20.0f, 2.35f, -5.5f, 0.3f, 4.7f, 0.3f);
    box(25.5f, 2.35f, -5.5f, 0.3f, 4.7f, 0.3f);
    box(31.0f, 2.35f, -5.5f, 0.3f, 4.7f, 0.3f);

    // sloping roof (front edge higher than the back)
    glPushMatrix();
    glTranslatef(cx, 4.2f, cz);
    glRotatef(-12.0f, 1.0f, 0.0f, 0.0f);
    color3(0.45f, 0.30f, 0.25f);
    box(0.0f, 0.0f, 0.0f, 12.0f, 0.25f, 6.2f);
    glPopMatrix();

    // hay bale in the corner
    color3(0.85f, 0.75f, 0.35f);
    box(30.0f, 0.45f, -9.6f, 1.2f, 0.9f, 1.0f);
}

// ---- A blocky person. It faces +z, then yaw turns it. ----
void drawPerson(float x, float z, float yaw, float scale,
                float legSwing, float armSwing, float armUp, float bob,
                const float* shirt, const float* pants, const float* hair, bool bags,
                bool sit)
{
    const float skin[3] = { 0.42f, 0.28f, 0.18f };

    // when sitting, lower the whole body so the thighs rest on the bench (top at y = 0.60)
    float rootY = bob + (sit ? (0.60f - 0.73f * scale) : 0.0f);

    glPushMatrix();
    glTranslatef(x, rootY, z);
    glRotatef(yaw, 0.0f, 1.0f, 0.0f);
    glScalef(scale, scale, scale);

    // legs (pivot at the hip)
    for (int side = -1; side <= 1; side += 2) {
        glPushMatrix();
        glTranslatef(0.15f * side, 0.85f, 0.0f);
        color3(pants[0], pants[1], pants[2]);
        if (sit) {
            box(0.0f,  0.00f, 0.25f, 0.22f, 0.24f, 0.50f);    // thigh points forward
            box(0.0f, -0.35f, 0.45f, 0.22f, 0.70f, 0.22f);    // lower leg hangs down
        } else {
            glRotatef(legSwing * side, 1.0f, 0.0f, 0.0f);
            box(0.0f, -0.425f, 0.0f, 0.22f, 0.85f, 0.24f);
        }
        glPopMatrix();
    }

    // torso
    color3(shirt[0], shirt[1], shirt[2]);
    box(0.0f, 1.25f, 0.0f, 0.6f, 0.8f, 0.34f);

    // arms (pivot at the shoulder). armUp is a raise angle, armSwing is the walk swing
    for (int side = -1; side <= 1; side += 2) {
        glPushMatrix();
        glTranslatef(0.38f * side, 1.58f, 0.0f);
        glRotatef(armSwing * side + armUp, 1.0f, 0.0f, 0.0f);
        color3(shirt[0], shirt[1], shirt[2]);
        box(0.0f, -0.35f, 0.0f, 0.16f, 0.7f, 0.16f);
        color3(skin[0], skin[1], skin[2]);
        box(0.0f, -0.74f, 0.0f, 0.14f, 0.14f, 0.14f);
        glPopMatrix();
    }

    // head, hair, face
    color3(skin[0], skin[1], skin[2]);
    box(0.0f, 1.88f, 0.0f, 0.40f, 0.40f, 0.40f);
    color3(hair[0], hair[1], hair[2]);
    box(0.0f, 2.10f, -0.02f, 0.44f, 0.14f, 0.44f);
    if (!shadowPass) {
        color3(0.05f, 0.05f, 0.05f);
        box(-0.09f, 1.93f, 0.205f, 0.06f, 0.06f, 0.02f);   // eyes
        box( 0.09f, 1.93f, 0.205f, 0.06f, 0.06f, 0.02f);
        color3(0.75f, 0.15f, 0.15f);
        box(0.0f, 1.80f, 0.205f, 0.14f, 0.03f, 0.02f);     // smile
    }

    // shopping bags
    if (bags) {
        color3(0.85f, 0.65f, 0.25f);
        box( 0.52f, 0.75f, 0.10f, 0.30f, 0.40f, 0.22f);
        color3(0.30f, 0.60f, 0.85f);
        box(-0.52f, 0.75f, 0.10f, 0.30f, 0.40f, 0.22f);
    }
    glPopMatrix();
}

// ---- A blocky car (faces +x). Wheels are tori with spokes so you see them turn. ----
void drawWheel(float wx, float wz, float wheelDeg)
{
    glPushMatrix();
    glTranslatef(wx, 0.42f, wz);
    glRotatef(-wheelDeg, 0.0f, 0.0f, 1.0f);     // rolling forward = clockwise seen from +z
    color3(0.08f, 0.08f, 0.08f);
    glutSolidTorus(0.12, 0.30, 8, 14);
    color3(0.65f, 0.65f, 0.68f);
    box(0.0f, 0.0f, 0.0f, 0.55f, 0.08f, 0.10f);   // spokes
    box(0.0f, 0.0f, 0.0f, 0.08f, 0.55f, 0.10f);
    glPopMatrix();
}

void drawCar(float x)
{
    float wheelDeg = x * (180.0f / PI) / 0.42f;       // rolls without slipping

    glPushMatrix();
    // Rotate the car along the same slanted direction as the road/driveway
    glRotatef(ROAD_ROT, 0.0f, 1.0f, 0.0f);
    glTranslatef(x, 0.0f, CAR_Z);

    color3(0.78f, 0.12f, 0.12f);
    box(0.0f, 0.80f, 0.0f, 4.2f, 0.80f, 1.8f);        // lower body
    color3(0.35f, 0.50f, 0.65f);
    box(-0.25f, 1.45f, 0.0f, 2.1f, 0.70f, 1.5f);      // cabin (glass)
    color3(0.78f, 0.12f, 0.12f);
    box(-0.25f, 1.85f, 0.0f, 2.2f, 0.10f, 1.6f);      // roof
    color3(0.20f, 0.20f, 0.22f);
    box(2.12f, 0.50f, 0.0f, 0.12f, 0.2f, 1.7f);       // bumpers
    box(-2.12f, 0.50f, 0.0f, 0.12f, 0.2f, 1.7f);
    color3(1.00f, 0.95f, 0.70f);
    box(2.12f, 0.90f,  0.55f, 0.06f, 0.2f, 0.3f);     // headlights
    box(2.12f, 0.90f, -0.55f, 0.06f, 0.2f, 0.3f);
    color3(0.80f, 0.05f, 0.05f);
    box(-2.12f, 0.90f,  0.55f, 0.06f, 0.2f, 0.3f);    // tail lights
    box(-2.12f, 0.90f, -0.55f, 0.06f, 0.2f, 0.3f);

    drawWheel( 1.3f,  0.95f, wheelDeg);
    drawWheel( 1.3f, -0.95f, wheelDeg);
    drawWheel(-1.3f,  0.95f, wheelDeg);
    drawWheel(-1.3f, -0.95f, wheelDeg);
    glPopMatrix();
}

// ---- All the people, with the pose that fits what they are doing ----
void drawPeople()
{
    const float dadShirt[3] = { 0.15f, 0.35f, 0.70f }, dadPants[3] = { 0.20f, 0.20f, 0.25f };
    const float mumShirt[3] = { 0.80f, 0.25f, 0.40f }, mumPants[3] = { 0.35f, 0.25f, 0.50f };
    const float k1Shirt[3]  = { 0.95f, 0.80f, 0.15f }, k1Pants[3]  = { 0.20f, 0.40f, 0.60f };
    const float k2Shirt[3]  = { 0.20f, 0.70f, 0.40f }, k2Pants[3]  = { 0.50f, 0.30f, 0.20f };
    const float hairCol[3]  = { 0.08f, 0.06f, 0.05f };

    const float* shirts[4] = { dadShirt, mumShirt, k1Shirt, k2Shirt };
    const float* pantsC[4] = { dadPants, mumPants, k1Pants, k2Pants };
    const float  scales[4] = { 1.00f, 0.92f, 0.60f, 0.50f };

    float t = timeOfDay;
    float walkPhase = timeOfDay * 880.0f;

    for (int i = 0; i < 4; i++) {
        const PersonState& p = story.p[i];
        if (p.hidden) continue;

        float legSwing = 0.0f, armSwing = 0.0f, armUp = 0.0f, bob = 0.0f;
        bool  kid = (i >= 2);

        if (p.moving) {
            legSwing = 35.0f * sinf(walkPhase + i);
            armSwing = 30.0f * sinf(walkPhase + i);
        } else if (kid && ((t > 0.11f && t < 0.33f) || (t > 0.37f && t < 0.43f))) {
            // cheering: arms up and jumping
            armUp = -150.0f + 15.0f * sinf(timeOfDay * 900.0f + i);
            bob   = 0.25f * scales[i] * fabsf(sinf(timeOfDay * 450.0f + i * 1.7f));
        } else if (i == 0 && t > 0.42f && t < 0.52f) {
            // dad cooking: arms forward, stirring
            armUp = -65.0f + 12.0f * sinf(timeOfDay * 700.0f);
        }

        // sitting on the bench at the picnic table
        bool sit = (!p.moving && p.x > 6.0f && p.z > 4.0f && p.z < 5.2f);
        if (sit) armUp = -55.0f + 20.0f * sinf(timeOfDay * 500.0f + i);   // eating

        // dad carries the shopping bags from the car to the barbecue, and later into the house
        bool bags = (i == 0 && ((t > 0.343f && t < 0.42f) || (t > 0.672f && t < 0.738f)));
        drawPerson(p.x, p.z, p.yaw, scales[i], legSwing, armSwing, armUp, bob,
                   shirts[i], pantsC[i], hairCol, bags, sit);
    }
}

// Everything that casts a shadow
void drawCasters()
{
    drawHouse();
    for (int i = 0; i < NUM_TREES; i++)
        drawTree(TREES[i].x, TREES[i].z, TREES[i].s);
    drawBush( -5.7f,   1.7f);
    drawBush( 11.5f,   6.5f);
    drawProps();

    if (story.carOn) drawCar(story.carX);

        drawCowshed();

    if (story.elephOn) {
    
                for (int i = 0; i < NUM_COWS; i++) {
            float t   = timeOfDay;
            float cx2 = cowXAt(i, t);
            drawCow(cx2, COW_Z[i], COW_SCALE[i], cx2 * 1.8f,
                    cowMoving(i, t) ? 1.0f : 0.0f, story.cowLie, i, cowYaw(i, t));
        }

    }

    drawPeople();
}

void drawGround()
{
    // The ground used to be ONE giant quad. Fog and lighting are worked out per
    // corner, so with corners 1000 units away (some behind the camera) the
    // fog colour smeared across the whole screen and the grass turned white or
    // flat depending on the camera angle. Now the ground is a grid of small
    // cells, so every cell gets a correct fog value and the colour is the same
    // from every angle.
    static std::vector<float> g;
    if (g.empty()) {
        const float outer[4] = { -1000.0f, -400.0f, -200.0f, -100.0f };
        for (int i = 0; i < 4; i++) g.push_back(outer[i]);
        for (float v = -80.0f; v <= 80.0f; v += 8.0f) g.push_back(v);
        for (int i = 3; i >= 0; i--) g.push_back(-outer[i]);
    }
    int n = (int)g.size();

    // patches of slightly different greens (same palette as the blades)
    const float dark[3]  = { 0.22f, 0.56f, 0.19f };
    const float light[3] = { 0.31f, 0.69f, 0.24f };

    glNormal3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_QUADS);
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            int ci[4] = { i, i,     i + 1, i + 1 };
            int cj[4] = { j, j + 1, j + 1, j     };
            for (int k = 0; k < 4; k++) {
                unsigned h = (unsigned)ci[k] * 374761393u + (unsigned)cj[k] * 668265263u;
                h = (h ^ (h >> 13)) * 1274126177u;
                h ^= (h >> 16);
                float t = (float)(h & 0xFFFF) / 65535.0f;
                // float c[3];
                // mix3(c, dark, light, t);
                // glColor3f(c[0], c[1], c[2]);
                // 
                glColor3f(0.27f, 0.65f, 0.22f);
                glVertex3f(g[ci[k]], 0.0f, g[cj[k]]);
            }
        }
    }
    glEnd();
}

void drawPath()                              // shorter dirt path from the door that only touches the road
{
    glPushMatrix();
    glRotatef(HOUSE_ROT, 0.0f, 1.0f, 0.0f);
    glColor3f(0.78f, 0.70f, 0.52f);
    glBegin(GL_QUADS);
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-0.9f, 0.012f,  3.6f);
    glVertex3f( 0.9f, 0.012f,  3.6f);
    // Shortened length to end right at the road edge (matching CAR_Z - 1.6f)
    glVertex3f( 0.9f, 0.012f,  CAR_Z - 1.6f);
    glVertex3f(-0.9f, 0.012f,  CAR_Z - 1.6f);
    glEnd();
    glPopMatrix();
}

void drawRoad()                              // road slanted in the same direction as the driveway
{
    glPushMatrix();
    glRotatef(ROAD_ROT, 0.0f, 1.0f, 0.0f);
    glColor3f(0.30f, 0.30f, 0.32f);
    glBegin(GL_QUADS);
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-60.0f, 0.010f, CAR_Z - 1.6f);
    glVertex3f(-60.0f, 0.010f, CAR_Z + 1.6f);
    glVertex3f( 60.0f, 0.010f, CAR_Z + 1.6f);
    glVertex3f( 60.0f, 0.010f, CAR_Z - 1.6f);
    glEnd();
    glPopMatrix();
}

// ---------------------------------------------------------------------
//  7. SMOKE: grey puffs that rise, grow and fade (chimney + barbecue)
// ---------------------------------------------------------------------
void drawSmokePuffs(float x, float y, float z, int count, float rise,
                    float maxSize, float drift, float cycles)
{
    float c = lerpf(0.85f, 0.35f, g_sun.night);
    for (int i = 0; i < count; i++) {
        float u = fmodf(timeOfDay * cycles + (float)i / (float)count, 1.0f);   // age 0..1
        float px = x + drift * u + 0.25f * sinf(u * 6.0f + i);
        float py = y + rise * u;
        float r  = maxSize * (0.3f + 0.7f * u);
        float a  = 0.55f * (1.0f - u) * clamp01(u * 8.0f);
        glColor4f(c, c, c, a);
        glPushMatrix();
        glTranslatef(px, py, z);
        glutSolidSphere(r, 10, 8);
        glPopMatrix();
    }
}

void drawSmoke()
{
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    float cx, cz;
    localToWorld(2.4f, -0.8f, cx, cz);                       // chimney position
    drawSmokePuffs(cx, 6.6f, cz, 6, 4.5f, 0.55f, 2.0f, 17.5f);

    if (story.grillOn)
        drawSmokePuffs(2.6f, 1.3f, 8.0f, 5, 3.0f, 0.35f, 1.0f, 25.0f);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

// ---------------------------------------------------------------------
//  8. LIGHTS AND SHADOWS
// ---------------------------------------------------------------------
void setupLights()
{
    const SunState& s = g_sun;
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_NORMALIZE);

    float amb[4]  = { s.ambient[0], s.ambient[1], s.ambient[2], 1.0f };
    float none[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb);

    // LIGHT0 = the sun (directional light, w = 0)
    float pos0[4] = { s.dir[0], s.dir[1], s.dir[2], 0.0f };
    float dif0[4] = { s.color[0] * s.intensity, s.color[1] * s.intensity,
                      s.color[2] * s.intensity, 1.0f };
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, pos0);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  dif0);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  none);
    glLightfv(GL_LIGHT0, GL_SPECULAR, none);

    // LIGHT1 = soft sky light / moonlight from the camera side
    float pos1[4] = { 0.3f, 0.6f, 1.0f, 0.0f };
    float dif1[4] = { s.fillCol[0], s.fillCol[1], s.fillCol[2], 1.0f };
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_POSITION, pos1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  dif1);
    glLightfv(GL_LIGHT1, GL_AMBIENT,  none);
    glLightfv(GL_LIGHT1, GL_SPECULAR, none);
}

// Matrix that flattens (projects) geometry onto a plane, seen from a light.
// plane = (a,b,c,d) with ax+by+cz+d = 0,  light = (x,y,z,w) (w = 0: directional)
void shadowMatrix(float m[16], const float plane[4], const float light[4])
{
    float dot = plane[0] * light[0] + plane[1] * light[1]
              + plane[2] * light[2] + plane[3] * light[3];

    m[0] = dot - light[0] * plane[0];
    m[4] = 0.0f - light[0] * plane[1];
    m[8] = 0.0f - light[0] * plane[2];
    m[12]= 0.0f - light[0] * plane[3];

    m[1] = 0.0f - light[1] * plane[0];
    m[5] = dot - light[1] * plane[1];
    m[9] = 0.0f - light[1] * plane[2];
    m[13]= 0.0f - light[1] * plane[3];

    m[2] = 0.0f - light[2] * plane[0];
    m[6] = 0.0f - light[2] * plane[1];
    m[10]= dot - light[2] * plane[2];
    m[14]= 0.0f - light[2] * plane[3];

    m[3] = 0.0f - light[3] * plane[0];
    m[7] = 0.0f - light[3] * plane[1];
    m[11]= 0.0f - light[3] * plane[2];
    m[15]= dot - light[3] * plane[3];
}

void drawShadows()
{
    const SunState& s = g_sun;

    // ground plane y = 0.02 (a tiny bit above the grass and road to avoid z-fighting)
    float plane[4] = { 0.0f, 1.0f, 0.0f, -0.02f };
    float light[4] = { s.dir[0], s.dir[1], s.dir[2], 0.0f };
    float m[16];
    shadowMatrix(m, plane, light);

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    // stencil: every ground pixel may be darkened only ONCE
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_EQUAL, 0, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);

    glColor4f(0.0f, 0.0f, 0.0f, s.shadowAlpha);

    glPushMatrix();
    glMultMatrixf(m);                        // flatten everything onto the ground
    shadowPass = true;
    drawCasters();
    //drawGrassArrays(false);                  // every blade casts its own thin shadow
    shadowPass = false;
    glPopMatrix();

    glDisable(GL_STENCIL_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

// ---------------------------------------------------------------------
//  9. 3D PASS, ZZZ and HUD
// ---------------------------------------------------------------------
void draw3D()
{
    const SunState& s = g_sun;

    glEnable(GL_DEPTH_TEST);

    // fog blends the far ground into the sky colour at the horizon
    // (tinted green so the far lawn never fades to white, and dimmed at night)
    float hd = (0.35f + 0.65f * s.vis) * (1.0f - 0.85f * s.night);
    const float hazeGreen[3] = { 0.30f * hd, 0.58f * hd, 0.26f * hd };
    float fogMix[3];
    mix3(fogMix, s.horizon, hazeGreen, 0.60f);
    float fogColor[4] = { fogMix[0], fogMix[1], fogMix[2], 1.0f };
    glHint(GL_FOG_HINT, GL_NICEST);
    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogf(GL_FOG_START, 90.0f);
    glFogf(GL_FOG_END,   750.0f);

    setupLights();

    drawGround();
    drawRoad();
    drawPath();
    if (showShadows && s.shadowAlpha > 0.01f) drawShadows();
    drawCasters();
    drawSmoke();
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
}

// "Z z Z" floating up from the bedroom window (2D text at the window's screen position)
void drawZzz(float zx, float zy, float strength, float phase)
{
    if (strength < 0.01f) return;
    begin2D();
    for (int k = 0; k < 3; k++) {
        float u  = fmodf(timeOfDay * 14.0f + phase + k / 3.0f, 1.0f);
        float px = zx + 8.0f + u * 55.0f;
        float py = zy + 10.0f + u * 95.0f;
        float a  = sinf(PI * u) * strength;
        void* font = GLUT_BITMAP_HELVETICA_12;
        char  ch   = 'z';
        if (u > 0.66f)      { font = GLUT_BITMAP_TIMES_ROMAN_24; ch = 'Z'; }
        else if (u > 0.33f) { font = GLUT_BITMAP_HELVETICA_18;   ch = 'Z'; }
        glColor4f(1.0f, 1.0f, 1.0f, a);
        glRasterPos2f(px, py);
        glutBitmapCharacter(font, ch);
    }
    end2D();
}

void drawHUD()
{
    const SunState& s = g_sun;
    begin2D();

    // text panel
    glColor4f(0.0f, 0.0f, 0.0f, 0.45f);
    glBegin(GL_QUADS);
    glVertex2f(10.0f,  (float)winH - 86.0f);
    glVertex2f(690.0f, (float)winH - 86.0f);
    glVertex2f(690.0f, (float)winH - 10.0f);
    glVertex2f(10.0f,  (float)winH - 10.0f);
    glEnd();

    float hrs = fmodf(4.8f + 24.0f * timeOfDay, 24.0f);      // clock time
    int hh = (int)hrs;
    int mm = (int)((hrs - hh) * 60.0f);
    if (mm > 59) mm = 59;
    const char* phase = "Night";
    if      (hrs < 5.5f)  phase = "Night";
    else if (hrs < 7.0f)  phase = "Sunrise";
    else if (hrs < 11.0f) phase = "Morning";
    else if (hrs < 13.0f) phase = "Midday";
    else if (hrs < 16.5f) phase = "Afternoon";
    else if (hrs < 19.0f) phase = "Sunset";

    char line[200];
    snprintf(line, sizeof(line), "A FAMILY DAY    Time %02d:%02d  (%s)    Shadows: %s%s",
             hh, mm, phase, showShadows ? "ON" : "OFF", paused ? "    [PAUSED]" : "");
    glColor3f(1.0f, 1.0f, 1.0f);
    drawText(20.0f, (float)winH - 30.0f, line);
    drawText(20.0f, (float)winH - 50.0f, eventText());
    drawText(20.0f, (float)winH - 72.0f,
             "SPACE pause   +/- speed   [ ] scrub   Arrows camera   H shadows   R reset   ESC quit");

    // timeline bar at the bottom (24 hours)
    float x0f = 40.0f, x1f = (float)winW - 40.0f;
    int x0 = (int)x0f, x1 = (int)x1f;
    int yb0 = 24, yb1 = 32;

    glColor4f(0.0f, 0.0f, 0.0f, 0.35f);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);          glVertex2f((float)winW, 0.0f);
    glVertex2f((float)winW, 46.0f);  glVertex2f(0.0f, 46.0f);
    glEnd();

    glColor4f(1.0f, 1.0f, 1.0f, 0.9f);
    glBegin(GL_POINTS);
    // rectangle outline: 4 Bresenham lines
    bresenhamLine(x0, yb0, x1, yb0);
    bresenhamLine(x1, yb0, x1, yb1);
    bresenhamLine(x1, yb1, x0, yb1);
    bresenhamLine(x0, yb1, x0, yb0);
    // hour ticks every 3 hours: DDA lines
    for (int hr = 0; hr < 24; hr += 3) {
        float f = fmodf((float)hr - 4.8f + 24.0f, 24.0f) / 24.0f;
        float x = x0f + f * (x1f - x0f);
        ddaLine(x, 16.0f, x, 23.0f);
    }
    glEnd();

    for (int hr = 0; hr < 24; hr += 3) {
        float f = fmodf((float)hr - 4.8f + 24.0f, 24.0f) / 24.0f;
        float x = x0f + f * (x1f - x0f);
        char lab[8];
        snprintf(lab, sizeof(lab), "%02d:00", hr);
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText(x - 15.0f, 3.0f, lab);
    }

    // marker on the bar: midpoint circle (sun colour by day, pale moon at night)
    int mx = (int)lroundf(x0f + timeOfDay * (x1f - x0f));
    int my = (yb0 + yb1) / 2;
    float mk[3];
    const float moonCol[3] = { 0.90f, 0.92f, 0.85f };
    mix3(mk, s.disc, moonCol, s.night);
    glColor4f(mk[0], mk[1], mk[2], 1.0f);
    glBegin(GL_LINES);
    midpointCircleFill(mx, my, 7);
    glEnd();
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    midpointCircle(mx, my, 8);
    glEnd();

    end2D();
}

// ---------------------------------------------------------------------
//  10. DISPLAY (the rendering pipeline in order)
// ---------------------------------------------------------------------
void display()
{
    updateSun();
    updateStory();
    const SunState& s = g_sun;

    glClearColor(s.horizon[0], s.horizon[1], s.horizon[2], 1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, winW, winH);            // viewport transformation

    // --- projection: viewing frustum ---
    float aspect = (float)winW / (float)winH;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV, aspect, 1.0, 4000.0);

    // --- view: camera orbits the house ---
    float eye[3] = { sinf(camYaw * DEG) * camDist, 3.0f, cosf(camYaw * DEG) * camDist };
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye[0], eye[1], eye[2],
              LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
              0.0, 1.0, 0.0);

    // --- use the pipeline matrices to find where things appear on screen ---
    GLdouble mv[16], pr[16];
    GLint    vp[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, mv);
    glGetDoublev(GL_PROJECTION_MATRIX, pr);
    glGetIntegerv(GL_VIEWPORT, vp);

    float fx = LOOK_AT[0] - eye[0];
    float fy = LOOK_AT[1] - eye[1];
    float fz = LOOK_AT[2] - eye[2];
    float fl3 = sqrtf(fx * fx + fy * fy + fz * fz);
    float fl2 = sqrtf(fx * fx + fz * fz);

    // horizon = a far away point at eye height straight ahead
    GLdouble hxw, hyw, hzw;
    gluProject(eye[0] + fx / fl2 * 3000.0, eye[1], eye[2] + fz / fl2 * 3000.0,
               mv, pr, vp, &hxw, &hyw, &hzw);
    float hy = (float)hyw;
    if (hy < 0.0f) hy = 0.0f;
    if (hy > (float)winH) hy = (float)winH;

    // sun = a far away point in the direction of the sun
    float facing = (fx * s.dir[0] + fy * s.dir[1] + fz * s.dir[2]) / fl3;
    bool  sunVisible = (facing > 0.1f);      // only if it is in front of the camera
    GLdouble sxw = 0.0, syw = 0.0, szw = 0.0;
    if (sunVisible)
        gluProject(eye[0] + s.dir[0] * 3000.0, eye[1] + s.dir[1] * 3000.0,
                   eye[2] + s.dir[2] * 3000.0, mv, pr, vp, &sxw, &syw, &szw);

    // moon = a fixed spot in the sky (azimuth -18, elevation 28)
    float maz = -18.0f * DEG, mel = 28.0f * DEG;
    float md[3] = { sinf(maz) * cosf(mel), sinf(mel), -cosf(maz) * cosf(mel) };
    float mfacing = (fx * md[0] + fy * md[1] + fz * md[2]) / fl3;
    bool  moonVisible = (mfacing > 0.1f);
    GLdouble mxw = 0.0, myw = 0.0, mzw = 0.0;
    if (moonVisible)
        gluProject(eye[0] + md[0] * 3000.0, eye[1] + md[1] * 3000.0,
                   eye[2] + md[2] * 3000.0, mv, pr, vp, &mxw, &myw, &mzw);

    // bedroom window (house side wall) -> screen position for the "zzz"
    float bwx, bwz;
    localToWorld(4.1f, 0.0f, bwx, bwz);
    GLdouble zxw = 0.0, zyw = 0.0, zzw = 0.0;
    gluProject(bwx, 2.9, bwz, mv, pr, vp, &zxw, &zyw, &zzw);

    // background shifts a little when the camera turns
    float hfov = 2.0f * atanf(tanf(FOV * 0.5f * DEG) * aspect) / DEG;
    float bgShift = camYaw * ((float)winW / hfov);

    drawBackground(hy, sunVisible, (float)sxw, (float)syw,
                   moonVisible, (float)mxw, (float)myw, bgShift);
    draw3D();
        drawZzz((float)zxw, (float)zyw, story.zzz, 0.0f);          // the family's bedroom

            for (int i = 0; i < NUM_COWS; i++) {                        // the sleeping cows
        GLdouble cxw = 0.0, cyw = 0.0, czw = 0.0;
        gluProject(COW_END[i], 1.3, COW_Z[i] + 1.4, mv, pr, vp, &cxw, &cyw, &czw);
        drawZzz((float)cxw, (float)cyw, story.cowZzz, i * 0.31f);
    }

        for (int i = 0; i < NUM_COWS; i++) {                        // the sleeping cows
        GLdouble cxw = 0.0, cyw = 0.0, czw = 0.0;
        gluProject(COW_END[i], 1.3, COW_Z[i] + 1.4, mv, pr, vp, &cxw, &cyw, &czw);
        drawZzz((float)cxw, (float)cyw, story.cowZzz, i * 0.31f);
    }
    drawHUD();

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    winW = (w > 1) ? w : 1;
    winH = (h > 1) ? h : 1;
}

// ---------------------------------------------------------------------
//  11. ANIMATION AND INPUT
// ---------------------------------------------------------------------
void timer(int)
{
    int now = glutGet(GLUT_ELAPSED_TIME);
    float dt = (now - lastTime) / 1000.0f;
    lastTime = now;
    if (dt > 0.1f) dt = 0.1f;

    if (!paused) {
        timeOfDay += daySpeed * dt;
        if (timeOfDay >= 1.0f) timeOfDay -= 1.0f;   // new day
    }

    float k = dt * 6.0f;                            // the door eases open / closed
    if (k > 1.0f) k = 1.0f;
    doorOpen += (story.doorTarget - doorOpen) * k;

    rayAngle += 0.25f * dt;                         // rays keep turning
    windTime += dt;                                 // grass keeps swaying (even when paused)
    for (int i = 0; i < 5; i++) {                   // clouds drift
        clouds[i].x += clouds[i].speed * dt;
        if (clouds[i].x > winW + 200.0f) clouds[i].x = -200.0f;
    }

    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);
}

void wrapTime()
{
    while (timeOfDay < 0.0f)  timeOfDay += 1.0f;
    while (timeOfDay >= 1.0f) timeOfDay -= 1.0f;
}

void keyboard(unsigned char key, int, int)
{
    switch (key) {
    case 27:  exit(0); break;
    case ' ': paused = !paused; break;
    case '+': case '=':
        daySpeed *= 1.25f;
        if (daySpeed > 0.5f) daySpeed = 0.5f;
        break;
    case '-': case '_':
        daySpeed /= 1.25f;
        if (daySpeed < 0.002f) daySpeed = 0.002f;
        break;
    case '[': timeOfDay -= 0.01f; wrapTime(); break;
    case ']': timeOfDay += 0.01f; wrapTime(); break;
    case 'h': case 'H': showShadows = !showShadows; break;
    case 'r': case 'R':
        timeOfDay = 0.0f; daySpeed = 1.0f / 70.0f; paused = false;
        camYaw = 0.0f; camDist = 24.0f; showShadows = true;
        break;
    }
}

void special(int key, int, int)
{
    switch (key) {
    case GLUT_KEY_LEFT:  camYaw  -= 3.0f; break;
    case GLUT_KEY_RIGHT: camYaw  += 3.0f; break;
    case GLUT_KEY_UP:    camDist -= 1.0f; break;
    case GLUT_KEY_DOWN:  camDist += 1.0f; break;
    }
    if (camYaw  < -40.0f) camYaw  = -40.0f;
    if (camYaw  >  40.0f) camYaw  =  40.0f;
    if (camDist <  10.0f) camDist =  10.0f;
    if (camDist >  50.0f) camDist =  50.0f;
}

// ---------------------------------------------------------------------
//  12. MAIN
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    glutInitWindowSize(winW, winH);
    glutCreateWindow("A Family Day - OpenGL");

    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    initStars();
    //initGrass();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);

    lastTime = glutGet(GLUT_ELAPSED_TIME);
    glutTimerFunc(16, timer, 0);

    printf("Controls: SPACE pause | + / - speed | [ ] scrub | arrows camera | H shadows | R reset | ESC quit\n");
    glutMainLoop();
    return 0;
}