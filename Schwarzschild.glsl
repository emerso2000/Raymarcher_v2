#version 450 core
#define PI 3.14159265359

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D screen;

layout(std140, binding = 1) uniform CameraBlock {
    vec3 cam_o;
    float padding1;
} camera;

const int MAX_STEPS = 1950;
const float b = 2.0;

float atan2(in float y, in float x) {
    return x == 0.0 ? sign(y)*PI/2 : atan(y, x);
}

vec3 cartesianToSpherical(vec3 cartesian) {
    float radius = length(cartesian); //make negative to test for other universe
    float theta = acos(cartesian.y / radius);
    float phi = -atan2(cartesian.z, cartesian.x);

    return vec3(radius, theta, phi);
}

vec3 sphericalToCartesian(vec4 spherical) {
    float x = spherical.x * sin(spherical.y) * cos(spherical.z);
    float y = spherical.x * cos(spherical.y);
    float z = spherical.x * sin(spherical.y) * sin(spherical.z);

    return vec3(x, y, z);
}

vec3 cartesianToAzELR(vec3 cartesianVec, vec3 newRayOrigin) {
    float r = newRayOrigin.x; //make negative to test for other universe
    float th = newRayOrigin.y;
    float phi = newRayOrigin.z;

    mat3 transformationMatrix = mat3(
        sin(th)*cos(phi),  sin(th)*sin(phi),  cos(th),
        cos(th)*cos(phi),  cos(th) *sin(phi), -sin(th),
        -sin(phi),  cos(phi),  0
    );
    
    vec3 newVec = transformationMatrix * cartesianVec;

    return newVec;
}

mat4 calculateChristoffelSymbolsAlphaR(vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    mat4 christoffelSymbols_alpha_r;

    christoffelSymbols_alpha_r[0][0] = 0.0;
    christoffelSymbols_alpha_r[0][1] = 0.0;
    christoffelSymbols_alpha_r[0][2] = 0.0;
    christoffelSymbols_alpha_r[0][3] = 0.0;

    christoffelSymbols_alpha_r[1][0] = 0.0;
    christoffelSymbols_alpha_r[1][1] = -r;
    christoffelSymbols_alpha_r[1][2] = 0.0;
    christoffelSymbols_alpha_r[1][3] = 0.0;

    christoffelSymbols_alpha_r[2][0] = 0.0;
    christoffelSymbols_alpha_r[2][1] = 0.0;
    christoffelSymbols_alpha_r[2][2] = -r * sin(theta) * sin(theta);
    christoffelSymbols_alpha_r[2][3] = 0.0;

    christoffelSymbols_alpha_r[3][0] = 0.0;
    christoffelSymbols_alpha_r[3][1] = 0.0;
    christoffelSymbols_alpha_r[3][2] = 0.0;
    christoffelSymbols_alpha_r[3][3] = 0.0;
    
    return christoffelSymbols_alpha_r;
}

mat4 calculateChristoffelSymbolsAlphaTheta(vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    mat4 christoffelSymbols_alpha_theta;

    christoffelSymbols_alpha_theta[0][0] = 0.0;
    christoffelSymbols_alpha_theta[0][1] = r / ((b * b) + (r * r));
    christoffelSymbols_alpha_theta[0][2] = 0.0;
    christoffelSymbols_alpha_theta[0][3] = 0.0;

    christoffelSymbols_alpha_theta[1][0] = r / ((b * b) + (r * r));
    christoffelSymbols_alpha_theta[1][1] = 0.0;
    christoffelSymbols_alpha_theta[1][2] = 0.0;
    christoffelSymbols_alpha_theta[1][3] = 0.0;

    christoffelSymbols_alpha_theta[2][0] = 0.0;
    christoffelSymbols_alpha_theta[2][1] = 0.0;
    christoffelSymbols_alpha_theta[2][2] = -sin(theta) * cos(theta);
    christoffelSymbols_alpha_theta[2][3] = 0.0;

    christoffelSymbols_alpha_theta[3][0] = 0.0;
    christoffelSymbols_alpha_theta[3][1] = 0.0;
    christoffelSymbols_alpha_theta[3][2] = 0.0;
    christoffelSymbols_alpha_theta[3][3] = 0.0;

    return christoffelSymbols_alpha_theta;
}

mat4 calculateChristoffelSymbolsAlphaPhi(vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    mat4 christoffelSymbols_alpha_phi;

    christoffelSymbols_alpha_phi[0][0] = 0.0;
    christoffelSymbols_alpha_phi[0][1] = 0.0;
    christoffelSymbols_alpha_phi[0][2] = r / ((b * b) + (r * r));
    christoffelSymbols_alpha_phi[0][3] = 0.0;

    christoffelSymbols_alpha_phi[1][0] = 0.0;
    christoffelSymbols_alpha_phi[1][1] = 0.0;
    christoffelSymbols_alpha_phi[1][2] = 1 / tan(theta);
    christoffelSymbols_alpha_phi[1][3] = 0.0;

    christoffelSymbols_alpha_phi[2][0] = r / ((b * b) + (r * r));
    christoffelSymbols_alpha_phi[2][1] = 1 / tan(theta);
    christoffelSymbols_alpha_phi[2][2] = 0.0;
    christoffelSymbols_alpha_phi[2][3] = 0.0;

    christoffelSymbols_alpha_phi[3][0] = 0.0;
    christoffelSymbols_alpha_phi[3][1] = 0.0;
    christoffelSymbols_alpha_phi[3][2] = 0.0;
    christoffelSymbols_alpha_phi[3][3] = 0.0;

    return christoffelSymbols_alpha_phi;
}

mat4 calculateChristoffelSymbolsAlphaTime(vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    mat4 christoffelSymbols_alpha_time;

    christoffelSymbols_alpha_time[0][0] = 0.0;
    christoffelSymbols_alpha_time[0][1] = 0.0;
    christoffelSymbols_alpha_time[0][2] = 0.0;
    christoffelSymbols_alpha_time[0][3] = 0.0;

    christoffelSymbols_alpha_time[1][0] = 0.0;
    christoffelSymbols_alpha_time[1][1] = 0.0;
    christoffelSymbols_alpha_time[1][2] = 0.0;
    christoffelSymbols_alpha_time[1][3] = 0.0;

    christoffelSymbols_alpha_time[2][0] = 0.0;
    christoffelSymbols_alpha_time[2][1] = 0.0;
    christoffelSymbols_alpha_time[2][2] = 0.0;
    christoffelSymbols_alpha_time[2][3] = 0.0;

    christoffelSymbols_alpha_time[3][0] = 0.0;
    christoffelSymbols_alpha_time[3][1] = 0.0;
    christoffelSymbols_alpha_time[3][2] = 0.0;
    christoffelSymbols_alpha_time[3][3] = 0.0;
    
    return christoffelSymbols_alpha_time;
}

vec3 marchRay(vec4 origin, vec4 direction) {
    float stepSize = 0.01;
    float wall = 8.0f;

    vec4 p = origin;

    vec4 accel = vec4(0.0);

    float ref_step = 0.01;

    for (int i = 0; i < MAX_STEPS; i++) {
        p += stepSize * direction.xyzw;
        vec3 p_cart = sphericalToCartesian(p);

        if(length(p_cart.xz) < 2.0) {
            stepSize = max(ref_step * length(p_cart.xz) / 0.5, 0.0001);
        }
        if (p.x < 0) {
            return vec3(1.0, 0.0, 0.0);
        }
        if (p_cart.z <= -wall) {
            return vec3(0.0, 1.0, 0.0); //first wall
        }
        if (p_cart.x >= wall) {
            return vec3(0.0, 0.0, 1.0); // Second wall
        }
        if (p_cart.x <= -wall) {
            return vec3(1.0, 1.0, 0.0); // Third wall
        }
        if (p_cart.z >= wall) {
            return vec3(0.0, 1.0, 1.0); // Wall behind the camera
        }
        if (p_cart.y <= -wall) {
            return vec3(0.5, 0.0, 0.0); //floor
        }
        if (p_cart.y >= wall) {
            return vec3(0.5, 0.5, 0.0); //ceiling
        }

        mat4 christoffelSymbols_alpha_r = calculateChristoffelSymbolsAlphaR(p);
        mat4 christoffelSymbols_alpha_theta = calculateChristoffelSymbolsAlphaTheta(p);
        mat4 christoffelSymbols_alpha_phi = calculateChristoffelSymbolsAlphaPhi(p);
        mat4 christoffelSymbols_alpha_time = calculateChristoffelSymbolsAlphaTime(p);

        // Calculate the accelerations using the geodesic equation
        accel.x = -dot(direction.xyzw, christoffelSymbols_alpha_r * direction.xyzw);
        accel.y = -dot(direction.xyzw, christoffelSymbols_alpha_theta * direction.xyzw);
        accel.z = -dot(direction.xyzw, christoffelSymbols_alpha_phi * direction.xyzw);
        accel.w = -dot(direction.xyzw, christoffelSymbols_alpha_time * direction.xyzw);

        direction.xyzw += accel * stepSize;
    }
    return vec3(0.115, 0.133, 0.173);
}

vec4 get_timelike_vector(vec3 cartesian_basis_speed, float time_direction, float e0, float e1, float e2, float e3) {
    float v = length(cartesian_basis_speed);
    float Y = 1.0 / sqrt(1.0 - v * v);

    float B = v;

    float psi = B * Y;

    vec4 e0vec = vec4(e0, 0, 0, 0);
    vec4 e1vec = vec4(0, e1, 0, 0);
    vec4 e2vec = vec4(0, 0, e2, 0);
    vec4 e3vec = vec4(0, 0, 0, e3);

    vec4 bT = time_direction * Y * e0vec;

    vec3 dir = normalize(cartesian_basis_speed);

    if (v == 0.0) {
        dir = vec3(0.0, 0.0, 1.0);
    }

    vec4 bX = psi * dir.x * e1vec;
    vec4 bY = psi * dir.y * e2vec;
    vec4 bZ = psi * dir.z * e3vec;

    return vec4(bT + bX + bY + bZ);
}

vec4 integrateGeodesic(vec4 currentPosition, vec4 currentVelocity) {
    float timeStep = 0.01;

    vec4 acceleration = vec4(0.0, 0.0, 0.0, 0.0);

    mat4 christoffelSymbols_alpha_r = calculateChristoffelSymbolsAlphaR(currentPosition);
    mat4 christoffelSymbols_alpha_theta = calculateChristoffelSymbolsAlphaTheta(currentPosition);
    mat4 christoffelSymbols_alpha_phi = calculateChristoffelSymbolsAlphaPhi(currentPosition);
    mat4 christoffelSymbols_alpha_time = calculateChristoffelSymbolsAlphaTime(currentPosition);

    // Calculate the accelerations using the geodesic equation
    acceleration.x = -dot(currentVelocity.xyzw, christoffelSymbols_alpha_r * currentVelocity.xyzw);
    acceleration.y = -dot(currentVelocity.xyzw, christoffelSymbols_alpha_theta * currentVelocity.xyzw);
    acceleration.z = -dot(currentVelocity.xyzw, christoffelSymbols_alpha_phi * currentVelocity.xyzw);
    acceleration.w = -dot(currentVelocity.xyzw, christoffelSymbols_alpha_time * currentVelocity.xyzw);

    currentVelocity.xyzw += acceleration * timeStep;

    return currentVelocity.xyzw;
}

void main() {
    vec4 pixel = vec4(0.115, 0.133, 0.173, 1.0);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    ivec2 dims = imageSize(screen);

    vec2 uv = (vec2(pixel_coords) - 0.5 * dims.xy) / dims.y;

    vec3 ro = camera.cam_o;
    vec3 rd = (vec3(uv.x, uv.y, 0.4));

    rd = normalize(rd);
    
    vec3 sphericalRo = cartesianToSpherical(ro);
    vec3 sphericalRd = cartesianToAzELR(rd, sphericalRo);

    vec4 ellisRd = vec4(sphericalRd, -1.0);
    vec4 ellisRo = vec4(sphericalRo, 0.0);
    
    float r = sphericalRo.x; //initial position

    float e0 = 1.0; //time
    float e1 = 1.0; //r
    float e2 = 1.0 / sqrt((b * b) + (sphericalRo.x * sphericalRo.x)); //theta
    float e3 = 1.0 / (sqrt((b * b) + (sphericalRo.x * sphericalRo.x)) * sin(sphericalRo.y)); //phi

    //in order of (r, theta, phi, ct)
    mat4 tetradMatrix = mat4(
        e1, 0, 0, 0,
        0, e2, 0, 0,
        0, 0, e3, 0,
        0, 0, 0, e0
    );    
    
    ellisRd = ellisRd * tetradMatrix; //transform to true schwarzschild coordinates

    vec4 initialPosition = vec4(0, camera.cam_o);
    vec4 initialVelocity = get_timelike_vector(vec3(0, 0, 1), 1.0, e0, e1, e2, e3);

    // for (int i = 0; i < MAX_STEPS; i++) {
    //     camera.cam_o = integrateGeodesic(initialPosition, initialVelocity).xyz;
    // }

    vec3 color = marchRay(ellisRo, ellisRd);

    pixel = vec4(color, 1.0);
    
    imageStore(screen, pixel_coords, pixel);
}