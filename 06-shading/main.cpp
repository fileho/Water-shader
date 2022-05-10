/*
 * Source code for the NPGR019 lab practices. Copyright Martin Kahoun 2021.
 * Licensed under the zlib license, see LICENSE.txt in the root directory.
 */

#include <cstdio>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include <MathSupport.h>
#include <Camera.h>
#include <Geometry.h>
#include <Textures.h>
#include <memory>

#include "shaders.h"
#include "Water.h"

// Set to 1 to create debugging context that reports errors, requires OpenGL 4.3!
#define _ENABLE_OPENGL_DEBUG 0

// ----------------------------------------------------------------------------
// GLM optional parameters:
// GLM_FORCE_LEFT_HANDED       - draw the left handed coordinate system
// GLM_FORCE_XYZW_ONLY         - simplify vector types and draw x, y, z, w only
// ----------------------------------------------------------------------------
// For remapping depth to [0, 1] interval draw GLM option below with glClipControl

//glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE); // requires version >= 4.5
// //
// GLM_FORCE_DEPTH_ZERO_TO_ONE - force projection depth mapping to [0, 1]
//                               must draw glClipControl(), requires OpenGL 4.5
//
// More information about the matter here:
// https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_clip_control.txt
// ----------------------------------------------------------------------------

// Structure for holding window parameters
struct Window
{
	// Window default width
	static const int DefaultWidth = 800;
	// Window default height
	static const int DefaultHeight = 600;

	// Width in pixels
	int width;
	// Height in pixels
	int height;
	// Main window handle
	GLFWwindow *handle;
} mainWindow = {0};

// ----------------------------------------------------------------------------

// Near clip plane settings
float nearClipPlane = 0.5f;
// Far clip plane settings
float farClipPlane = 150.1f;
// Camera FOV
float fov = 45.0f;

// ----------------------------------------------------------------------------

// Mouse movement
struct MouseStatus
{
	// Current position
	double x, y;
	// Previous position
	double prevX, prevY;

	// Updates the status - called once per frame
	void Update(double &moveX, double &moveY)
	{
		moveX = x - prevX;
		prevX = x;
		moveY = y - prevY;
		prevY = y;
	}
} mouseStatus = {0.0};

// ----------------------------------------------------------------------------

// Camera movement speeds
static constexpr float CameraNormalSpeed = 5.0f;
static constexpr float CameraTurboSpeed = 50.0f;

// ----------------------------------------------------------------------------

// Maximum number of allowed instances - must match the instancing vertex shader!
static const unsigned int MAX_INSTANCES = 1024;
// Max buffer length
static const unsigned int MAX_TEXT_LENGTH = 256;
// MSAA samples
static const GLsizei MSAA_SAMPLES = 4;
// Used MSAA samples
GLsizei msaaLevel = MSAA_SAMPLES;

// Number of cubes in the scene
const int numCubes = 9;
// Cube position
std::vector<glm::vec3> cubePositions;


// Camera instance
Camera camera;
// Quad instance
Mesh<Vertex_Pos_Nrm_Tgt_Tex> *quad = nullptr;
// Cube instance
Mesh<Vertex_Pos_Nrm_Tgt_Tex> *cube = nullptr;
// Water
std::unique_ptr<Water> water = nullptr;

// Textures helper instance
Textures& textures(Textures::GetInstance());

// General draw VAO
GLuint vao = 0;
// Our framebuffer object
GLuint fbo = 0;
// Our render target for rendering
GLuint renderTarget = 0;
// Our depth stencil for rendering
GLuint depthStencil = 0;

// Vsync on?
bool vsync = true;
// Depth test on?
bool depthTest = true;
// Draw wireframe?
bool wireframe = false;
// Tonemapping on?
bool tonemapping = true;
// Instancing buffer handle
GLuint instancingBuffer = 0;
// Transformation matrices uniform buffer object
GLuint transformBlockUBO = 0;

// Data for a single object instance
struct InstanceData
{
	// In this simple example just a transformation matrix, transposed for efficient storage
	glm::mat3x4 transformation;
};

// ----------------------------------------------------------------------------

// Textures we'll be using
namespace LoadedTextures
{
	enum
	{
		White, Grey, Blue, CheckerBoard, Diffuse, Normal, Specular, Occlusion, BumpMap, NumTextures
	};
}
GLuint loadedTextures[LoadedTextures::NumTextures] = {0};

// ----------------------------------------------------------------------------

// Forward declaration for the framebuffer creation
void createFramebuffer(int width, int height, GLsizei MSAA);

// Callback for handling GLFW errors
void errorCallback(int error, const char* description)
{
	printf("GLFW Error %i: %s\n", error, description);
}

#if _ENABLE_OPENGL_DEBUG
void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam)
{
	if (type == GL_DEBUG_TYPE_ERROR)
		printf("OpenGL error: %s\n", message);
}
#endif

// Callback for handling window resize events
void resizeCallback(GLFWwindow* window, int width, int height)
{
	mainWindow.width = width;
	mainWindow.height = height;
	glViewport(0, 0, width, height);
	camera.SetProjection(fov, (float)width / (float)height, nearClipPlane, farClipPlane);
	if (water)
		water->set_resolution(width, height);

	createFramebuffer(width, height, msaaLevel);
}

// Callback for handling mouse movement over the window - called when mouse movement is detected
void mouseMoveCallback(GLFWwindow* window, double x, double y)
{
	// Update the current position
	mouseStatus.x = x;
	mouseStatus.y = y;
}

// Keyboard callback for handling system switches
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// Notify the window that user wants to exit the application
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	// Enable/disable MSAA - note that it still uses the MSAA buffer
	if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
	{
		if (msaaLevel > 1)
		{
			msaaLevel = 1;
		}
		else
		{
			msaaLevel = MSAA_SAMPLES;
		}

		createFramebuffer(mainWindow.width, mainWindow.height, msaaLevel);
	}

	// Enable/disable wireframe rendering
	if (key == GLFW_KEY_F2 && action == GLFW_PRESS)
	{
		wireframe = !wireframe;
	}

	// Enable/disable backface culling
	if (key == GLFW_KEY_F3 && action == GLFW_PRESS)
	{
		if (glIsEnabled(GL_CULL_FACE))
			glDisable(GL_CULL_FACE);
		else
			glEnable(GL_CULL_FACE);
	}

	// Enable/disable depth test
	if (key == GLFW_KEY_F4 && action == GLFW_PRESS)
	{
		depthTest = !depthTest;
	}

	// Enable/disable vsync
	if (key == GLFW_KEY_F5 && action == GLFW_PRESS)
	{
		vsync = !vsync;
		if (vsync)
			glfwSwapInterval(1);
		else
			glfwSwapInterval(0);
	}

	// Enable/disable vsync
	if (key == GLFW_KEY_F6 && action == GLFW_PRESS)
	{
		tonemapping = !tonemapping;
	}

	// Zoom in
	if (key == GLFW_KEY_KP_ADD || key == GLFW_KEY_EQUAL && action == GLFW_PRESS)
	{
		fov -= 1.0f;
		if (fov < 5.0f)
			fov = 5.0f;
	}

	// Zoom out
	if (key == GLFW_KEY_KP_SUBTRACT || key == GLFW_KEY_MINUS && action == GLFW_PRESS)
	{
		fov += 1.0f;
		if (fov >= 180.0f)
			fov = 179.0f;
	}

	if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS)
	{
		fov = 45.0f;
	}

	// Set the camera projection
	camera.SetProjection(fov, (float)mainWindow.width / (float)mainWindow.height, nearClipPlane, farClipPlane);
}

// ----------------------------------------------------------------------------

// Helper method for loading textures
void loadTextures()
{
	// Create texture samplers
	textures.CreateSamplers();

	// Prepare textures
	loadedTextures[LoadedTextures::White] = Textures::CreateSingleColorTexture(255, 255, 255);
	loadedTextures[LoadedTextures::Grey] = Textures::CreateSingleColorTexture(127, 127, 127);
	loadedTextures[LoadedTextures::Blue] = Textures::CreateSingleColorTexture(127, 127, 255);
	loadedTextures[LoadedTextures::CheckerBoard] = Textures::CreateCheckerBoardTexture(256, 16);
	loadedTextures[LoadedTextures::Diffuse] = Textures::LoadTexture("data/Terracotta_Tiles_002_Base_Color.jpg", true);
	loadedTextures[LoadedTextures::Normal] = Textures::LoadTexture("data/Terracotta_Tiles_002_Normal.jpg", false);
	loadedTextures[LoadedTextures::Specular] = Textures::LoadTexture("data/Terracotta_Tiles_002_Roughness.jpg", false);
	loadedTextures[LoadedTextures::Occlusion] = Textures::LoadTexture("data/Terracotta_Tiles_002_ambientOcclusion.jpg", false);
	loadedTextures[LoadedTextures::BumpMap] = Textures::LoadTexture("data/water_bump.jpg", false);
}

// Helper method for creating scene geometry
void createGeometry()
{
	// Create general draw VAO
	glGenVertexArrays(1, &vao);

	// Prepare meshes
	quad = Geometry::CreateQuadNormalTangentTex();
	cube = Geometry::CreateCubeNormalTangentTex();
	water = std::make_unique<Water>();
	

	{
		// Generate the instancing buffer as Uniform Buffer Object
		glGenBuffers(1, &instancingBuffer);
		glBindBuffer(GL_UNIFORM_BUFFER, instancingBuffer);

		// Obtain UBO index and size from the instancing shader program
		GLuint uboIndex = glGetUniformBlockIndex(shaderProgram[ShaderProgram::Instancing], "InstanceBuffer");
		GLint uboSize = 0;
		glGetActiveUniformBlockiv(shaderProgram[ShaderProgram::Instancing], uboIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &uboSize);

		// Describe the buffer data - we're going to change this every frame
		glBufferData(GL_UNIFORM_BUFFER, uboSize, nullptr, GL_DYNAMIC_DRAW);

		// Unbind the GL_UNIFORM_BUFFER target for now
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	{
		// Generate the transform UBO handle
		glGenBuffers(1, &transformBlockUBO);
		glBindBuffer(GL_UNIFORM_BUFFER, transformBlockUBO);

		// Obtain UBO index from the default shader program:
		// we're gonna bind this UBO for all shader programs and we're making
		// assumption that all of the UBO's used by our shader programs are
		// all the same size
		GLuint uboIndex = glGetUniformBlockIndex(shaderProgram[ShaderProgram::Default], "TransformBlock");
		GLint uboSize = 0;
		glGetActiveUniformBlockiv(shaderProgram[ShaderProgram::Default], uboIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &uboSize);

		// Describe the buffer data - we're going to change this every frame
		glBufferData(GL_UNIFORM_BUFFER, uboSize, nullptr, GL_DYNAMIC_DRAW);

		// Bind the memory for usage
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, transformBlockUBO);

		// Unbind the GL_UNIFORM_BUFFER target for now
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	cubePositions.reserve(numCubes);

	// Generate random positions for the rest of the cubes
	const float offset{ 30 };
	for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++)
		{
			if (!i && !j)
				continue;
			cubePositions.emplace_back(i * offset, -14.0f, j * offset);
		}
	}
	cubePositions.emplace_back(20, 8, 4);
}

// Helper method for OpenGL initialization
bool initOpenGL()
{
	// Set the GLFW error callback
	glfwSetErrorCallback(errorCallback);

	// Initialize the GLFW library
	if (!glfwInit()) return false;

	// Request OpenGL 3.3 core profile upon window creation
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_SAMPLES, 0); // Disable MSAA, we'll handle it ourselves
#if _ENABLE_OPENGL_DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create the window
	mainWindow.handle = glfwCreateWindow(Window::DefaultWidth, Window::DefaultHeight, "", nullptr, nullptr);
	if (mainWindow.handle == nullptr)
	{
		printf("Failed to create the GLFW window!");
		return false;
	}

	// Make the created window with OpenGL context current for this thread
	glfwMakeContextCurrent(mainWindow.handle);

	// Check that GLAD .dll loader and symbol imported is ready
	if (!gladLoadGL()) {
		printf("GLAD failed!\n");
		return false;
	}

#if _ENABLE_OPENGL_DEBUG
	// Enable error handling callback function - context must be created with DEBUG flags
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(debugCallback, nullptr);
	GLuint unusedIds = 0;
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, &unusedIds, true);
#endif

	// Check for available UBO size in bytes
	GLint maxUboSize;
	const GLint expectedUboSize = 4096 * 4 * 4;
	glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxUboSize);
	if (maxUboSize < expectedUboSize)
	{
		printf("Implementation allowed UBO size: %d B smaller than expected (%d B)!", maxUboSize, expectedUboSize);
		return false;
	}

	// Prevent crashes
	if (numCubes > MAX_INSTANCES)
	{
		printf("Trying to render more than the maximum number of cubes: %d!", MAX_INSTANCES);
		return false;
	}

	// Enable vsync
	if (vsync)
		glfwSwapInterval(1);
	else
		glfwSwapInterval(0);

	// Enable backface culling
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

	// Register a window resize callback
	glfwSetFramebufferSizeCallback(mainWindow.handle, resizeCallback);

	// Register keyboard callback
	glfwSetKeyCallback(mainWindow.handle, keyCallback);

	// Register mouse movement callback
	glfwSetCursorPosCallback(mainWindow.handle, mouseMoveCallback);

	// Set the OpenGL viewport and camera projection
	resizeCallback(mainWindow.handle, Window::DefaultWidth, Window::DefaultHeight);

	// Set the initial camera position and orientation
	camera.SetTransformation(glm::vec3(-3.0f, 3.0f, -5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	
	return true;
}

// Helper function for creating the HDR framebuffer
void createFramebuffer(int width, int height, GLsizei MSAA)
{
	// Bind the default framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Generate the FBO if necessary
	if (!fbo)
	{
		glGenFramebuffers(1, &fbo);
	}

	// Bind it and recreate textures
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// --------------------------------------------------------------------------
	// Render target texture:
	// --------------------------------------------------------------------------

	// Delete it if necessary
	if (glIsTexture(renderTarget))
	{
		glDeleteTextures(1, &renderTarget);
		renderTarget = 0;
	}

	// Create the texture name
	if (renderTarget == 0)
	{
		glGenTextures(1, &renderTarget);
	}

	// Bind and recreate the render target texture
	if (MSAA > 1)
	{
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderTarget);
		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, MSAA, GL_RGB16F, width, height, GL_TRUE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, renderTarget, 0);
	}
	else
	{
		glBindTexture(GL_TEXTURE_2D, renderTarget);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTarget, 0);
	}

	// --------------------------------------------------------------------------
	// Depth buffer texture:
	// --------------------------------------------------------------------------

	// Delete it if necessary
	if (glIsRenderbuffer(depthStencil))
	{
		glDeleteRenderbuffers(1, &depthStencil);
		depthStencil = 0;
	}

	// Create the depth-stencil name
	if (depthStencil == 0)
	{
		glGenRenderbuffers(1, &depthStencil);
	}

	// Bind and recreate the depth-stencil Render Buffer Object
	glBindRenderbuffer(GL_RENDERBUFFER, depthStencil);
	if (MSAA > 1)
	{
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, MSAA, GL_DEPTH_COMPONENT32F, width, height);
	}
	else
	{
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	}
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthStencil);

	// Set the list of draw buffers.
	GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, drawBuffers);

	// Check for completeness
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		printf("Failed to create framebuffer: 0x%04X\n", status);
	}

	// Bind back the window system provided framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Helper method for graceful shutdown
void shutDown()
{
	// Release shader programs
	for (int i = 0; i < ShaderProgram::NumShaderPrograms; ++i)
	{
		glDeleteProgram(shaderProgram[i]);
	}

	// Delete meshes
	delete quad;
	quad = nullptr;
	delete cube;
	cube = nullptr;

	// Release the instancing buffer
	glDeleteBuffers(1, &instancingBuffer);

	// Release the framebuffer
	glDeleteTextures(1, &renderTarget);
	glDeleteTextures(1, &depthStencil);
	glDeleteFramebuffers(1, &fbo);

	// Release the generic VAO
	glDeleteVertexArrays(1, &vao);

	// Release textures
	glDeleteTextures(LoadedTextures::NumTextures, loadedTextures);

	// Release the window
	glfwDestroyWindow(mainWindow.handle);

	// Close the GLFW library
	glfwTerminate();
}

// ----------------------------------------------------------------------------

// Helper method for handling input events
void processInput(float dt)
{
	// Camera movement - keyboard events
	int direction = (int)MovementDirections::None;
	if (glfwGetKey(mainWindow.handle, GLFW_KEY_W) == GLFW_PRESS)
		direction |= (int)MovementDirections::Forward;

	if (glfwGetKey(mainWindow.handle, GLFW_KEY_S) == GLFW_PRESS)
		direction |= (int)MovementDirections::Backward;

	if (glfwGetKey(mainWindow.handle, GLFW_KEY_A) == GLFW_PRESS)
		direction |= (int)MovementDirections::Left;

	if (glfwGetKey(mainWindow.handle, GLFW_KEY_D) == GLFW_PRESS)
		direction |= (int)MovementDirections::Right;

	if (glfwGetKey(mainWindow.handle, GLFW_KEY_R) == GLFW_PRESS)
		direction |= (int)MovementDirections::Up;

	if (glfwGetKey(mainWindow.handle, GLFW_KEY_F) == GLFW_PRESS)
		direction |= (int)MovementDirections::Down;

	// Camera speed
	if (glfwGetKey(mainWindow.handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		camera.SetMovementSpeed(CameraTurboSpeed);
	else
		camera.SetMovementSpeed(CameraNormalSpeed);

	// Update the mouse status
	double dx, dy;
	mouseStatus.Update(dx, dy);

	// Camera orientation - mouse movement
	glm::vec2 mouseMove(0.0f, 0.0f);
	if (glfwGetMouseButton(mainWindow.handle, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
	{
		mouseMove.x = (float)(dx);
		mouseMove.y = (float)(dy);
	}

	// Update the camera movement
	camera.Move((MovementDirections)direction, mouseMove, dt);

	// Reset camera state
	if (glfwGetKey(mainWindow.handle, GLFW_KEY_ENTER) == GLFW_PRESS)
	{
		camera.SetProjection(fov, (float)mainWindow.width / (float)mainWindow.height, nearClipPlane, farClipPlane);
		camera.SetTransformation(glm::vec3(-3.0f, 3.0f, -5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}
}

// Helper function for binding the appropriate textures
void bindTextures(const GLuint &diffuse, const GLuint &normal, const GLuint &specular, const GLuint &occlusion)
{
	// We want to bind textures and appropriate samplers
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, diffuse);
	glBindSampler(0, textures.GetSampler(Sampler::Anisotropic));

	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, normal);
	glBindSampler(1, textures.GetSampler(Sampler::Anisotropic));

	glActiveTexture(GL_TEXTURE0 + 2);
	glBindTexture(GL_TEXTURE_2D, specular);
	glBindSampler(2, textures.GetSampler(Sampler::Anisotropic));

	glActiveTexture(GL_TEXTURE0 + 3);
	glBindTexture(GL_TEXTURE_2D, occlusion);
	glBindSampler(3, textures.GetSampler(Sampler::Anisotropic));
}

// Helper function for creating and updating the instance data
void updateInstanceData()
{
	// Create transformation matrix
	static glm::mat4x4 transformation;
	// Instance data CPU side buffer
	static std::vector<InstanceData> instanceData(MAX_INSTANCES);

	// Cubes
	float angle = 20.0f;
	for (int i = 0; i < numCubes; ++i)
	{
		transformation = glm::translate(cubePositions[i]);
		transformation *= glm::scale(glm::vec3(30,30,30));
		instanceData[i].transformation = glm::transpose(transformation);
	}

	{
		int index = numCubes - 1;
		transformation = glm::translate(cubePositions[index]);
		transformation *= glm::rotate(glm::radians(angle), glm::vec3(1.0f, 1.0f, 1.0f));
		transformation *= glm::scale(glm::vec3(5, 5, 5));
		instanceData[index].transformation = glm::transpose(transformation);
	}
	

	// Bind the whole instancing buffer to the index 1
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, instancingBuffer);

	// Update the buffer data using mapping
	void *ptr = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
	memcpy(ptr, &*(instanceData.begin()), numCubes * sizeof(InstanceData));
	glUnmapBuffer(GL_UNIFORM_BUFFER);
}

void updateProgramData(GLuint program, const glm::vec3 &lightPosition)
{
	// TODO: make this a transform block as well

	// Update the light position
	GLint lightLoc = glGetUniformLocation(program, "lightPosWS");
	glUniform3f(lightLoc, lightPosition.x, lightPosition.y, lightPosition.z);

	// Update the view position
	GLint viewPosLoc = glGetUniformLocation(program, "viewPosWS");
	glm::vec4 viewPos = camera.GetViewToWorld()[3];
	glUniform4f(viewPosLoc, viewPos.x, viewPos.y, viewPos.z, viewPos.w);
}

// Helper method to update transformation uniform block
void updateTransformBlock(glm::mat4x3 cameraWorldToView = camera.GetWorldToView())
{
	// Tell OpenGL we want to work with our transform block
	glBindBuffer(GL_UNIFORM_BUFFER, transformBlockUBO);

	// Note: we should properly obtain block members size and offset via
	// glGetActiveUniformBlockiv() with GL_UNIFORM_SIZE, GL_UNIFORM_OFFSET,
	// I'm yoloing it here...

	// Update the world to view transformation matrix - transpose to 3 columns, 4 rows for storage in an uniform block:
	// per std140 layout column matrix CxR is stored as an array of C columns with R elements, i.e., 4x3 matrix would
	// waste space because it would require padding to vec4
	glm::mat3x4 worldToView = glm::transpose(cameraWorldToView);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat3x4), static_cast<const void*>(&*glm::value_ptr(worldToView)));

	// Update the projection matrix
	glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat3x4), sizeof(glm::mat4x4), static_cast<const void*>(&*glm::value_ptr(camera.GetProjection())));

	// Unbind the GL_UNIFORM_BUFFER target for now
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void draw_geometry(glm::vec4 clippingPlane = glm::vec4(0,0,0,0))
{
	// Light position
	static glm::vec3 lightPosition(-3.0f, 8.0f, 25.0f);

	// Draw the scene floor:
	{
		glUseProgram(shaderProgram[ShaderProgram::Default]);
		updateProgramData(shaderProgram[ShaderProgram::Default], lightPosition);

		// Create transformation matrix - 4 columns, 3 rows, last (0, 0, 0, 1) implicit to save space
		glm::mat4x3 transformation = glm::translate(glm::scale(glm::vec3(100.0f, 1.0f, 100.0f)), glm::vec3(0, -12, 0));
		glUniformMatrix4x3fv(0, 1, GL_FALSE, glm::value_ptr(transformation));

		bindTextures(loadedTextures[LoadedTextures::CheckerBoard], loadedTextures[LoadedTextures::Blue], loadedTextures[LoadedTextures::Grey], loadedTextures[LoadedTextures::White]);

		glBindVertexArray(quad->GetVAO());
		glDrawElements(GL_TRIANGLES, quad->GetIBOSize(), GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
	}

	// --------------------------------------------------------------------------
	
	// Draw cubes:
	{
		glUseProgram(shaderProgram[ShaderProgram::Instancing]);

		// Update the transformation & projection matrices
		updateProgramData(shaderProgram[ShaderProgram::Instancing], lightPosition);

		// Update instances and bind the instancing buffer
		updateInstanceData();
		glUniform4fv(0, 1, glm::value_ptr(clippingPlane));

		bindTextures(loadedTextures[LoadedTextures::Diffuse], loadedTextures[LoadedTextures::Normal], loadedTextures[LoadedTextures::Specular], loadedTextures[LoadedTextures::Occlusion]);

		glBindVertexArray(cube->GetVAO());
		glDrawElementsInstanced(GL_TRIANGLES, cube->GetIBOSize(), GL_UNSIGNED_INT, reinterpret_cast<void*>(0), numCubes);

		// Unbind the instancing buffer
		glBindBufferBase(GL_UNIFORM_BUFFER, 1, 0);
	}

	// --------------------------------------------------------------------------

	// Draw the light:
	{
		glUseProgram(shaderProgram[ShaderProgram::PointRendering]);

		// Update the light position
		GLint loc = glGetUniformLocation(shaderProgram[ShaderProgram::PointRendering], "position");
		glUniform3fv(loc, 1, glm::value_ptr(lightPosition));

		// Update the color
		loc = glGetUniformLocation(shaderProgram[ShaderProgram::PointRendering], "color");
		glUniform3f(loc, 1.0f, 1.0f, 1.0f);

		glPointSize(10.0f);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, 1);
	}
}

void renderScene()
{
	updateTransformBlock();

	// Bind the framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// Enable/disable depth test and write
	if (depthTest)
	{
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glDepthMask(GL_TRUE);
	}
	else
	{
		glDisable(GL_DEPTH_TEST);
	}

	// Enable/disable MSAA rendering
	if (msaaLevel > 1)
		glEnable(GL_MULTISAMPLE);
	else
		glDisable(GL_MULTISAMPLE);

	// Enable/disable wireframe rendering
	if (wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// Clear the color and depth buffer
	glClearColor(0.1f, 0.2f, 0.4f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// --------------------------------------------------------------------------

	
	glEnable(GL_CLIP_DISTANCE0);
	// refraction map
	water->render_refractions();
	draw_geometry();

	// reflexion map
	water->render_reflexions();
	updateTransformBlock(camera.PlanarReflexion());
	const glm::vec4 clippingPlane = glm::vec4(0, 1, 0, 0.8f);
	draw_geometry(clippingPlane);

	// main render	
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glViewport(0, 0, mainWindow.width, mainWindow.height);
	glEnable(GL_MULTISAMPLE);
	glClearColor(0.1f, 0.2f, 0.4f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	updateTransformBlock();

	draw_geometry();
	water->draw(camera.GetViewToWorld()[3]);

	// --------------------------------------------------------------------------

	// Unbind the shader program and other resources
	glBindVertexArray(0);
	glUseProgram(0);

	if (tonemapping)
	{
		// Unbind the framebuffer and bind the window system provided FBO
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Solid fill always
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		// Disable multisampling and depth test
		glDisable(GL_MULTISAMPLE);
		glDisable(GL_DEPTH_TEST);

		// Clear the color
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Tonemapping
		glUseProgram(shaderProgram[ShaderProgram::Tonemapping]);

		// Send in the required data
		glUniform1f(0, (float)msaaLevel);

		// Bind the HDR render target as texture
		GLenum target = (msaaLevel > 1) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(target, renderTarget);
		glBindSampler(0, 0); // Very important!

		// Draw fullscreen quad
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		// Unbind the shader program and other resources
		glBindVertexArray(0);
		glUseProgram(0);
	}
	else
	{
		// Just copy the render target to the screen
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		glDrawBuffer(GL_BACK);
		glBlitFramebuffer(0, 0, mainWindow.width, mainWindow.height, 0, 0, mainWindow.width, mainWindow.height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
	}
}

// Helper method for implementing the application main loop
void mainLoop()
{
	static double prevTime = 0.0;
	while (!glfwWindowShouldClose(mainWindow.handle))
	{
		// Calculate delta time
		double time = glfwGetTime();
		float dt = (float)(time - prevTime);
		prevTime = time;

		// Print it to the title bar
		static char title[MAX_TEXT_LENGTH];
		snprintf(title, MAX_TEXT_LENGTH, "dt = %.2fms, FPS = %.1f", dt * 1000.0, 1.0f / dt);
		glfwSetWindowTitle(mainWindow.handle, title);

		// Poll the events like keyboard, mouse, etc.
		glfwPollEvents();

		// Process keyboard input
		processInput(dt);

		// Render the scene
		renderScene();

		// Swap actual buffers on the GPU
		glfwSwapBuffers(mainWindow.handle);
	}
}

int main()
{
	// Initialize the OpenGL context and create a window
	if (!initOpenGL())
	{
		printf("Failed to initialize OpenGL!\n");
		shutDown();
		return -1;
	}

	// Compile shaders needed to run
	if (!compileShaders())
	{
		printf("Failed to compile shaders!\n");
		shutDown();
		return -1;
	}

	// Create the scene geometry
	createGeometry();

	// Load & create texture
	loadTextures();

	// Set Water shader
	water->set_program(shaderProgram[ShaderProgram::Water]);
	water->set_bump_map(loadedTextures[LoadedTextures::BumpMap]);
	glPatchParameteri(GL_PATCH_VERTICES, 4);

	// Enter the application main loop
	mainLoop();

	// Release used resources and exit
	shutDown();
	return 0;
}
