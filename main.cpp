#include </home/ethan/glad/glad.h>
#include <GLFW/glfw3.h>
#include </home/ethan/glm-stable/glm/glm.hpp>
#include </home/ethan/glm-stable/glm/gtc/type_ptr.hpp>
#include </home/ethan/glm-stable/glm/gtc/random.hpp>
#include <iostream>
#include </home/ethan/glm-stable/glm/gtc/matrix_transform.hpp>
#include <random>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <array>

#define PI 3.14159265359

const unsigned int SCREEN_WIDTH = 1920;
const unsigned int SCREEN_HEIGHT = 1080;

const float b = 2.0;
const int totalSteps = 600;

bool vSync = true;

GLfloat vertices[] = {
	-1.0f, -1.0f , 0.0f, 0.0f, 0.0f,
	-1.0f,  1.0f , 0.0f, 0.0f, 1.0f,
	 1.0f,  1.0f , 0.0f, 1.0f, 1.0f,
	 1.0f, -1.0f , 0.0f, 1.0f, 0.0f,
};

GLuint indices[] = {
	0, 2, 1,
	0, 3, 2
};

GLchar *LoadShader(const std::string &file)
{
	std::ifstream shaderFile;
	long shaderFileLength;

	shaderFile.open(file);

	if (shaderFile.fail())
	{
		throw std::runtime_error("COULD NOT FIND SHADER FILE");
	}

	shaderFile.seekg(0, shaderFile.end);
	shaderFileLength = shaderFile.tellg();
	shaderFile.seekg(0, shaderFile.beg);

	GLchar *shaderCode = new GLchar[shaderFileLength + 1];
	shaderFile.read(shaderCode, shaderFileLength);

	shaderFile.close();

	shaderCode[shaderFileLength] = '\0';

	return shaderCode;
}

struct CameraData {
    glm::vec3 cam_o;
    float padding1;
} camera;

void debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    std::cerr << "OpenGL debug message: " << message << std::endl;
}

glm::mat4 calculateChristoffelSymbolsAlphaR(glm::vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    glm::mat4 christoffelSymbols_alpha_r;

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

glm::mat4 calculateChristoffelSymbolsAlphaTheta(glm::vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    glm::mat4 christoffelSymbols_alpha_theta;

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

glm::mat4 calculateChristoffelSymbolsAlphaPhi(glm::vec4 position) { //issue here r is 0;
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    glm::mat4 christoffelSymbols_alpha_phi;

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

glm::mat4 calculateChristoffelSymbolsAlphaTime(glm::vec4 position) {
    //r theta phi ct

    float r = position.x;
    float theta = position.y;

    glm::mat4 christoffelSymbols_alpha_time;

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

glm::vec4 get_timelike_vector(glm::vec3 cartesian_basis_speed, float time_direction, float e0, float e1, float e2, float e3) {
    float v = glm::length(cartesian_basis_speed);
    float Y = 1.0 / sqrt(1.0 - v * v);

    float B = v;

    float psi = B * Y;

    glm::vec4 e0vec = glm::vec4(e0, 0, 0, 0);
    glm::vec4 e1vec = glm::vec4(0, e1, 0, 0);
    glm::vec4 e2vec = glm::vec4(0, 0, e2, 0);
    glm::vec4 e3vec = glm::vec4(0, 0, 0, e3);

    glm::vec4 bT = time_direction * Y * e0vec;

    glm::vec3 dir = glm::normalize(cartesian_basis_speed);

    if (v == 0.0) {
        dir = glm::vec3(0.0, 0.0, 1.0);
    }

    glm::vec4 bX = psi * dir.x * e1vec;
    glm::vec4 bY = psi * dir.y * e2vec;
    glm::vec4 bZ = psi * dir.z * e3vec;

    return glm::vec4(bT + bX + bY + bZ);
}

std::array<glm::vec4, totalSteps> integrateGeodesic(glm::vec4 currentPosition, glm::vec4 currentVelocity) {
    float timeStep = 0.01;

    std::array<glm::vec4, totalSteps> vectorArray;

    glm::vec4 acceleration = glm::vec4(0.0, 0.0, 0.0, 0.0);

	for (int i = 0; i < totalSteps; i++) {
		glm::mat4 christoffelSymbols_alpha_r = calculateChristoffelSymbolsAlphaR(currentPosition);
		glm::mat4 christoffelSymbols_alpha_theta = calculateChristoffelSymbolsAlphaTheta(currentPosition);
		glm::mat4 christoffelSymbols_alpha_phi = calculateChristoffelSymbolsAlphaPhi(currentPosition); //issue with infinity as value in this with initial condtion as (0, 0, 0)
		glm::mat4 christoffelSymbols_alpha_time = calculateChristoffelSymbolsAlphaTime(currentPosition);

		// Calculate the accelerations using the geodesic equation
		acceleration.x = -glm::dot(currentVelocity, christoffelSymbols_alpha_r * currentVelocity);
		acceleration.y = -glm::dot(currentVelocity, christoffelSymbols_alpha_theta * currentVelocity);
		acceleration.z = -glm::dot(currentVelocity, christoffelSymbols_alpha_phi * currentVelocity);
		acceleration.w = -glm::dot(currentVelocity, christoffelSymbols_alpha_time * currentVelocity);

		// std::cout << "Acceleration: (" << acceleration.x << ", "
		//           << acceleration.y << ", " << acceleration.z << ", "
		//           << acceleration.w << ")" << std::endl;

		currentVelocity += acceleration * timeStep;
		currentPosition += currentVelocity * timeStep;

        vectorArray[i] = currentPosition;  // Store the current position vector in the array
	}
    return vectorArray;
}

float atan2(float y, float x) {
    return x == 0.0 ? glm::sign(y)*PI/2 : glm::atan(y, x);
}

glm::vec3 cartesianToSpherical(glm::vec3 cartesian) {
    float radius = glm::length(cartesian); //make negative to test for other universe
    float theta = acos(cartesian.y / radius);
    float phi = -atan2(cartesian.z, cartesian.x);

    return glm::vec3(radius, theta, phi);
}

int main()
{
	camera.cam_o = glm::vec3(0.001f, 0.001f, -6.5f); 

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Ellis Wormhole", nullptr, nullptr);

	if (!window)
	{
		std::cout << "Failed to create the GLFW window\n";
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(vSync);

	gladLoadGL();

	GLuint VAO, VBO, EBO;
	glCreateVertexArrays(1, &VAO);
	glCreateBuffers(1, &VBO);
	glCreateBuffers(1, &EBO);

	glNamedBufferData(VBO, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glNamedBufferData(EBO, sizeof(indices), indices, GL_STATIC_DRAW);

	glEnableVertexArrayAttrib(VAO, 0);
	glVertexArrayAttribBinding(VAO, 0, 0);
	glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);

	glEnableVertexArrayAttrib(VAO, 1);
	glVertexArrayAttribBinding(VAO, 1, 0);
	glVertexArrayAttribFormat(VAO, 1, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat));

	glVertexArrayVertexBuffer(VAO, 0, VBO, 0, 5 * sizeof(GLfloat));
	glVertexArrayElementBuffer(VAO, EBO);

	GLuint screenTex;
	glCreateTextures(GL_TEXTURE_2D, 1, &screenTex);
	glTextureParameteri(screenTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(screenTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTextureParameteri(screenTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(screenTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureStorage2D(screenTex, 1, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT);
	glBindImageTexture(0, screenTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

	const GLchar *vertCode = LoadShader("../shader_files/shader.vert");
	const GLchar *fragCode = LoadShader("../shader_files/shader.frag");
	const GLchar *computeCode = LoadShader("../shader_files/Schwarzschild.glsl");

	GLuint screenVertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(screenVertexShader, 1, &vertCode, NULL);
	glCompileShader(screenVertexShader);

	GLuint screenFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(screenFragmentShader, 1, &fragCode, NULL);
	glCompileShader(screenFragmentShader);

	GLuint screenShaderProgram = glCreateProgram();
	glAttachShader(screenShaderProgram, screenVertexShader);
	glAttachShader(screenShaderProgram, screenFragmentShader);
	glLinkProgram(screenShaderProgram);

	GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(computeShader, 1, &computeCode, NULL);
	glCompileShader(computeShader);

	GLuint computeProgram = glCreateProgram();
	glAttachShader(computeProgram, computeShader);
	glLinkProgram(computeProgram);

	GLint success;
	glGetProgramiv(computeProgram, GL_LINK_STATUS, &success);

	if (success == GL_FALSE) {
		// Linking failed, retrieve the error log
		GLint logLength;
		glGetProgramiv(computeProgram, GL_INFO_LOG_LENGTH, &logLength);

		std::vector<GLchar> log(logLength);
		glGetProgramInfoLog(computeProgram, logLength, nullptr, log.data());

		// Output the error log
		std::cout << "Shader linking failed:\n" << log.data() << std::endl;
	} else {
		// Linking successful
		std::cout << "Shader linked successfully!" << std::endl;
	}

	//camera ubo
	unsigned int uboCameraBlock;
	glGenBuffers(1, &uboCameraBlock);
	glBindBuffer(GL_UNIFORM_BUFFER, uboCameraBlock);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraData), NULL, GL_STATIC_DRAW);

	glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboCameraBlock); 
	
	glm::vec3 sphericalRo = cartesianToSpherical(camera.cam_o);

	float e0 = 1.0; //time
    float e1 = 1.0; //r
    float e2 = 1.0 / sqrt((b * b) + (sphericalRo.x * sphericalRo.x)); //theta
    float e3 = 1.0 / (sqrt((b * b) + (sphericalRo.x * sphericalRo.x)) * sin(sphericalRo.y)); //phi

	glm::vec4 initialPosition = glm::vec4(0, camera.cam_o);
	glm::vec4 initialVelocity = get_timelike_vector(glm::vec3(0.5, 0.2, 0.1), 1.0, e0, e1, e2, e3);
	
	std::cout << "Initial Position: (" << initialPosition.x << ", "
              << initialPosition.y << ", " << initialPosition.z << ", "
              << initialPosition.w << ")" << std::endl;

	std::cout << "Initial Velocity: (" << initialVelocity.x << ", "
              << initialVelocity.y << ", " << initialVelocity.z << ", "
              << initialVelocity.w << ")" << std::endl;

	// camera.cam_o = integrateGeodesic(initialPosition, initialVelocity);
	std::array<glm::vec4, totalSteps> positionArray;

	positionArray = integrateGeodesic(initialPosition, initialVelocity);

	for (const auto& position : positionArray) {
        std::cout << "(" << position.x << ", " << position.y << ", " << position.z << ", " << position.w << ")" << std::endl;
    }

	std::cout << "Camera position: " << camera.cam_o.x << ", " << camera.cam_o.y << ", " << camera.cam_o.z << std::endl;			

	int currentIndex = 0;
	bool replay = true;

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

		if (currentIndex < totalSteps) {
			camera.cam_o = positionArray[currentIndex];
			currentIndex++;
		} 
		else {
			replay = false;
		}

        glBindBuffer(GL_UNIFORM_BUFFER, uboCameraBlock);
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, uboCameraBlock);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CameraData), &camera);

		glUseProgram(computeProgram);
		glDispatchCompute(std::ceil(SCREEN_WIDTH / 8), std::ceil(SCREEN_HEIGHT / 4), 1);
		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		glUseProgram(screenShaderProgram);
		glBindTextureUnit(0, screenTex);
		glUniform1i(glGetUniformLocation(screenShaderProgram, "screen"), 0);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(indices[0]), GL_UNSIGNED_INT, 0);

		// std::cout << "Camera position: " << camera.cam_o.x << ", " << camera.cam_o.y << ", " << camera.cam_o.z << std::endl;		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteShader(screenVertexShader);
	glDeleteShader(screenFragmentShader);
	glDeleteShader(computeProgram);

	glfwDestroyWindow(window);
	glfwTerminate();
}