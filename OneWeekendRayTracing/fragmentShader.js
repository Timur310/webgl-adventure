export default `#version 300 es
precision highp float;

#define PI 3.1415926538
#define MAX_FLOAT 1e5

out vec4 fragColor;
uvec4 s0;

uniform vec2 u_resolution;
uniform int u_frame;

void rng_initialize(vec2 p, int frame)
{
    //white noise seed
    s0 = uvec4(p, uint(frame), uint(p.x) + uint(p.y));
    
    //blue noise seed
    // s0 = uvec4(frame, frame*15843, frame*31 + 4566, frame*2345 + 58585);
}

void pcg4d( inout uvec4 v ) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  v = v ^ ( v >> 16u );
  v.x += v.y*v.w;
  v.y += v.z*v.x;
  v.z += v.x*v.y;
  v.w += v.y*v.z;
}

float rand() {
  pcg4d(s0);
  return float( s0.x ) / float( 0xffffffffu );
}
vec2 rand2() {
  pcg4d( s0 );
  return vec2( s0.xy ) / float(0xffffffffu);
}
vec3 rand3() {
  pcg4d(s0);
  return vec3( s0.xyz ) / float( 0xffffffffu );
}

vec3 randDirection() {
  vec2 r = rand2();
  float u = ( r.x - 0.5 ) * 2.0;
  float t = r.y * PI * 2.0;
  float f = sqrt( 1.0 - u * u );
  return vec3( f * cos( t ), f * sin( t ), u );
}

vec3 random_in_hemisphere(vec3 normal) {
  vec3 in_unit_sphere = randDirection();
  if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
      return in_unit_sphere;
  else
      return -in_unit_sphere;
}

vec3 random_unit_disk() {
  while (true) {
    vec3 p = vec3(rand(), rand(), 0);
    if (pow(length(p), 2.0) >= 1.0) continue;
    return p;
  }
}

struct Ray {
  vec3 origin;
  vec3 direction;
};

vec3 ray_at(float t, Ray r) {
  return r.origin + t * r.direction;
}

struct Cam {
  vec3 origin;
  vec3 horizontal;
  vec3 vertical;
  vec3 lower_left_corner;
  float lens_radius;
  vec3 u;
  vec3 v;
};

Cam create_camera() {
  // user specific settings
  vec3 lookfrom = vec3(-2,2,1);
  vec3 lookat = vec3(0,0,-1);
  vec3 vup = vec3(0,1,0);
  float aperture = 0.0;
  float focus_dist = length(lookfrom-lookat);

  float vfov = 90.0;
  float aspect_ratio = u_resolution.x / u_resolution.y;
  float theta = radians(vfov);
  float h = tan(theta/2.0);
  float viewport_height = 2.0 * h;
  float viewport_width = aspect_ratio * viewport_height;

  vec3 w = normalize(lookfrom - lookat);
  vec3 u = normalize(cross(vup, w));
  vec3 v = cross(w, u);

  vec3 origin = lookfrom;
  vec3 horizontal = focus_dist * viewport_width * u;
  vec3 vertical = focus_dist * viewport_height * v;
  vec3 lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - focus_dist*w;

  float lens_radius = aperture / 2.0;
  return Cam(origin,horizontal,vertical,lower_left_corner, lens_radius,u,v);
}

Ray get_cam_ray(vec2 uv, Cam cam) {
  vec3 rd = cam.lens_radius * random_unit_disk();
  vec3 offset = cam.u * rd.x + cam.v * rd.y;
  return Ray(cam.origin + offset,cam.lower_left_corner + uv.x*cam.horizontal + uv.y*cam.vertical - cam.origin - offset);
}

struct Mat {
  vec3 color;
  int type;
  float fuzz;
  float ir;
};

struct Hit_record {
  vec3 p;
  vec3 normal;
  float t;
  bool front_face;
  Mat matPointer;
};

void set_face_normal(Ray r, vec3 outward_normal, out Hit_record rec) {
  rec.front_face = dot(r.direction, outward_normal) < 0.0;
  rec.normal = rec.front_face ? outward_normal : -outward_normal;
}

struct Sphere {
  vec3 center;
  float radius;
  Mat mat;
};

bool sphere_hit(Ray r, Sphere s, float t_min, float t_max, inout Hit_record rec) {
  vec3 oc = r.origin - s.center;
  float a = pow(length(r.direction),2.0);
  float half_b = dot(oc, r.direction);
  float c = pow(length(oc),2.0) - s.radius*s.radius;

  float discriminant = half_b*half_b - a*c;
  if(discriminant < 0.0) return false;
  float sqrtd = sqrt(discriminant);

  float root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
      root = (-half_b + sqrtd) / a;
      if (root < t_min || t_max < root)
          return false;
  }

  rec.t = root;
  rec.p = ray_at(rec.t, r);
  rec.normal = (rec.p - s.center) / s.radius;
  vec3 outward_normal = (rec.p - s.center) / s.radius;
  rec.matPointer = s.mat;
  set_face_normal(r, outward_normal, rec);

  return true;
}


bool intersect_scene(const in Ray r, vec2 t_min_max, inout Hit_record h){
  //TODO: maybe a scene_generator class? with some random generation?!
  Sphere spheres[4];

  Mat gm = Mat(vec3(0.8, 0.8, 0.0),0,0.0,0.0); // ground
  Mat m = Mat(vec3(0.7, 0.3, 0.3),0,0.0,0.0); // lambertian
  Mat m2 = Mat(vec3(0.8, 0.6, 0.2),2,0.0,1.5); // dielectric
  Mat m3 = Mat(vec3(0.8, 0.8, 0.8),1,0.5,0.0); // fuzzy metal

  spheres[0] = Sphere(vec3(0,-100.5,-1), 100.0, gm);
  spheres[1] = Sphere(vec3(1,0,-1), 0.5,m2);
  spheres[2] = Sphere(vec3(-1,0,-1), 0.5,m3);
  spheres[3] = Sphere(vec3(0,0,-1), 0.5,m);

  // for(int i=1;i<3;i++) {
  //   Mat m = Mat(vec3(0.7, 0.3, 0.3));
  //   spheres[i] = Sphere(vec3(i,0,-1), 0.5,m);
  // }

  bool is_hit = false;
  
  // manually hit test ground plane for preventing z-fight
  // is_hit = sphere_hit(r, spheres[0], t_min_max.x,t_min_max.y, h) || is_hit;
  
  // begin to test the array of spheres
  for(int i=0;i<spheres.length();i++) {
    is_hit = sphere_hit(r, spheres[i], t_min_max.x,t_min_max.y, h) || is_hit;
  }
  
  return is_hit;
}

bool near_zero(vec3 e) {
  // Return true if the vector is close to zero in all dimensions.
  const float s = 1e-8;
  return abs(e.x) < s && abs(e.y) < s && abs(e.z) < s;
}

vec3 refract(vec3 uv, vec3 n, float etai_over_etat) {
  float cos_theta = min(dot(-uv, n), 1.0);
  vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
  vec3 r_out_parallel = -sqrt(abs(1.0 - pow(length(r_out_perp),2.0) )) * n;
  return r_out_perp + r_out_parallel;
}

float reflectance(float cosine, float ref_idx) {
  // Use Schlick's approximation
  float r0 = (1.0-ref_idx) / (1.0+ref_idx);
  r0 = r0*r0;
  return r0 + (1.0-r0)*pow((1.0 - cosine),5.0);
}

bool lambertian_scatter(Ray r, inout Hit_record rec, inout vec3 color, inout Ray scattered) {
  vec3 scatter_direction = rec.normal + randDirection();
  if (near_zero(scatter_direction)) {
    scatter_direction = rec.normal;
  }
  scattered = Ray(rec.p, scatter_direction);
  color *= rec.matPointer.color;
  return true;
}

bool metal_scatter(Ray r, inout Hit_record rec, inout vec3 color, inout Ray scattered) {
  vec3 reflected = reflect(normalize(r.direction), rec.normal);
  scattered = Ray(rec.p, reflected + rec.matPointer.fuzz * randDirection());
  color *= rec.matPointer.color;
  return (dot(scattered.direction, rec.normal) > 0.0);
}

bool dielectric_scatter(Ray r, inout Hit_record rec, inout vec3 color, inout Ray scattered) {
  color = vec3(1.0, 1.0, 1.0);
  float refraction_ratio = rec.front_face ? (1.0/rec.matPointer.ir) : rec.matPointer.ir;
  vec3 unit_direction = normalize(r.direction);
  float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
  float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
  bool cannot_refract = refraction_ratio * sin_theta > 1.0;
  vec3 direction;
  if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand()) {
    direction = reflect(unit_direction, rec.normal);
  }
  else {
    direction = refract(unit_direction, rec.normal, refraction_ratio);
  }
  scattered = Ray(rec.p, direction);
  return true;
}

vec3 ray_color(Ray r) {
  Hit_record rec;
  vec2 t_min_max = vec2(0.001, MAX_FLOAT);
  int depth = 10;

  Ray r_in;
  r_in.origin = r.origin;
  r_in.direction = r.direction;
  
  vec3 col = vec3(1.0);
  
  for(int i=0;i<depth;i++) {
    bool is_hit = intersect_scene(r_in, t_min_max, rec);
    Ray scattered;
    if(is_hit) {
      // lambertian
      if(rec.matPointer.type == 0) {
        if(lambertian_scatter(r_in, rec, col, scattered)) {
          r_in = scattered;
        }
      }
      // metal
      else if(rec.matPointer.type == 1) {
        if(metal_scatter(r_in, rec, col, scattered)) {
          r_in = scattered;
        }
      } else if(rec.matPointer.type == 2) {
        if(dielectric_scatter(r_in, rec, col, scattered)) {
          r_in = scattered;
        }
      } else {
        col = vec3(0);
      }
    } else {
      vec3 unit_direction = normalize(r_in.direction);
      float t = 0.5 * (unit_direction.y + 1.0);
      col *= (1.0-t) * vec3(1.0) + t * vec3(0.5,0.7,1.0);
      return col;
    }
  }
  return col;
}

vec3 color_correction(vec3 color, int samples) {
  float r = color.x;
  float g = color.y;
  float b = color.z;

  float scale = 1.0 / float(samples);
  r = sqrt(scale * r);
  g = sqrt(scale * g);
  b = sqrt(scale * b);
  return vec3(r,g,b);
}

void main() {
  rng_initialize(gl_FragCoord.xy,u_frame);
  Cam cam = create_camera();
  int samples = 50;
  vec3 outColor = vec3(0,0,0);
  
  for(int i=0;i<samples;++i) {
    float u = (gl_FragCoord.x + rand()) / u_resolution.x;
    float v = (gl_FragCoord.y + rand()) / u_resolution.y;

    Ray r = get_cam_ray(vec2(u,v), cam);

    outColor += ray_color(r);
  }
  vec3 final_color = color_correction(outColor, samples);
  fragColor = vec4(final_color, 1.0);
}
`;