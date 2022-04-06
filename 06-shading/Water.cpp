#include "Water.h"

#include <iostream>

#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"
#include "Textures.h"


Water::Water() :
	mesh_{ Geometry::CreateTessQuad(32) },
	model_to_world_{ scale(glm::vec3(30, 1, 30)) },
	start_time_{ clock::now() }
{
	refractions_.create();
	reflexions_.create();
}

// release the memory
Water::~Water() noexcept
{
	delete mesh_;
}

void Water::set_program(GLuint program)
{
	program_ = program;
}

void Water::render_refractions() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, refractions_.fbo);
	glDisable(GL_MULTISAMPLE);
	glClearColor(.1f, .1f, .7f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Water::render_reflexions() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, reflexions_.fbo);
	glDisable(GL_MULTISAMPLE);
	glClearColor(.1f, .1f, .3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Water::draw(const glm::vec4& viewPosWS)
{
	glUseProgram(program_);
	// set uniforms
	glUniformMatrix3x4fv(0, 1, GL_FALSE, value_ptr(model_to_world_));
	glUniform1f(1, get_time());
	glUniform4fv(2, 1, value_ptr(viewPosWS));
	
	glm::vec4 wave[] = 
	{
		glm::vec4(3.0f * tau_, 0, 1, 0.2f),
		glm::vec4(0, 9.3f * tau_, 0.6, 0.2f),
		glm::vec4(-1.8f * tau_, 2.0f * tau_, 1.0f, 0.1f),
		glm::vec4(1.6f * tau_, 3.2f * tau_, 1.1f, 0.1f),
		
	};
	glUniform4fv(3, 4, &(wave[0].x));

	// bind textures
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, refractions_.texture);
	glBindSampler(0, Textures::GetInstance().GetSampler(Sampler::Bilinear));
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, reflexions_.texture);
	glBindSampler(1, Textures::GetInstance().GetSampler(Sampler::Bilinear));
	// bind vao & draw
	glBindVertexArray(mesh_->GetVAO());
	glDrawElements(GL_TRIANGLES, mesh_->GetIBOSize(), GL_UNSIGNED_INT, nullptr);
}

void Water::render_target::create_ms(GLuint width, GLuint height)
{
	const GLsizei msaa{ 1 };
	
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	
	// create a color attachment texture
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, msaa, GL_RGB16F, width, height, GL_TRUE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, texture, 0);

	// create a depth render buffer
	glGenRenderbuffers(1, &depth);
	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, GL_DEPTH_COMPONENT32F, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

	// Set the list of draw buffers.
	GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, drawBuffers);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Water::render_target::create(GLuint width, GLuint height)
{
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

	glGenRenderbuffers(1, &depth);
	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

	GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, drawBuffers);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Water::render_target::~render_target() noexcept
{
	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &texture);
	glDeleteRenderbuffers(1, &depth);
}

float Water::get_time() const
{
	const time_t t = clock::now();
	const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(t - start_time_);
	return static_cast<float>(static_cast<double>(delta.count()) / 500.0);
}


