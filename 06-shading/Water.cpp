#include "Water.h"

#include <iostream>

#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/transform.hpp"
#include "Textures.h"


Water::Water() :
	mesh_{ Geometry::CreateQuad() },
	model_to_world_{ scale(glm::vec3(30, 1, 30)) },
	start_time_{ clock::now() }
{
	refractions_.create_depth(reflection_resolution());
	reflections_.create(refraction_resolution());
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
	const auto size = refraction_resolution();
	glViewport(0, 0, size.x, size.y);
	
	glDisable(GL_MULTISAMPLE);
	glClearColor(.1f, .1f, .7f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Water::render_reflexions() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, reflections_.fbo);
	const auto size = reflection_resolution();
	glViewport(0, 0, size.x, size.y);
	
	glDisable(GL_MULTISAMPLE);
	glClearColor(.1f, .3f, .7f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Water::draw(const glm::vec4& viewPosWS)
{
	glUseProgram(program_);
	// set uniforms
	glUniformMatrix3x4fv(0, 1, GL_FALSE, value_ptr(model_to_world_));
	glUniform1f(1, get_time());
	glUniform2f(2, static_cast<GLfloat>(resolution_.x), static_cast<GLfloat>(resolution_.y));
	glUniform4fv(3, 1, value_ptr(viewPosWS));
	
	glm::vec4 wave[] = 
	{
		glm::vec4(3.0f * tau_, 0, 1, 0.1f),
		glm::vec4(0, 1.3f * tau_, 0.6, 0.1f),
		glm::vec4(-1.8f * tau_, 2.0f * tau_, 1.0f, 0.05f),
		glm::vec4(1.6f * tau_, 3.2f * tau_, 1.1f, 0.05f),
		
	};
	glUniform4fv(4, 4, &(wave[0].x));

	// bind textures
	glActiveTexture(GL_TEXTURE0); // refraction map
	glBindTexture(GL_TEXTURE_2D, refractions_.texture);
	glBindSampler(0, Textures::GetInstance().GetSampler(Sampler::Bilinear));
	glActiveTexture(GL_TEXTURE1); // refraction map
	glBindTexture(GL_TEXTURE_2D, reflections_.texture);
	glBindSampler(1, Textures::GetInstance().GetSampler(Sampler::Bilinear));
	glActiveTexture(GL_TEXTURE2); // depth map
	glBindTexture(GL_TEXTURE_2D, refractions_.depth);
	glBindSampler(2, Textures::GetInstance().GetSampler(Sampler::Nearest));
	glActiveTexture(GL_TEXTURE3); // bump map
	glBindTexture(GL_TEXTURE_2D, bump_map_);
	glBindSampler(3, Textures::GetInstance().GetSampler(Sampler::Bilinear));
	// bind vao & draw
	glBindVertexArray(mesh_->GetVAO());

	glDrawArrays(GL_PATCHES, 0, 4);
	//glDrawElements(GL_TRIANGLES, mesh_->GetIBOSize(), GL_UNSIGNED_INT, nullptr);
}

void Water::set_resolution(int width, int height)
{
	resolution_ = glm::vec2(width, height);

	refractions_.create_depth(refraction_resolution());
	reflections_.create(reflection_resolution());
}

void Water::set_quality(float lod)
{
	quality_ = lod;

	refractions_.create_depth(refraction_resolution());
	reflections_.create(reflection_resolution());
}

void Water::render_target::create_depth(vec2i resolution)
{
	if (fbo == 0)
		glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	if (glIsTexture(texture))
		glDeleteTextures(1, &texture);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution.x, resolution.y, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

	if (glIsTexture(depth))
		glDeleteTextures(1, &depth);
	glGenTextures(1, &depth);
	glBindTexture(GL_TEXTURE_2D, depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, resolution.x, resolution.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);


	GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, drawBuffers);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Water::render_target::create(vec2i resolution)
{
	if (fbo == 0)
		glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	if (glIsTexture(texture))
	{
		glDeleteTextures(1, &texture);
	}
	glGenTextures(1, &texture);

	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution.x, resolution.y, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

	if (glIsRenderbuffer(depth))
		glDeleteRenderbuffers(1, &depth);
	
	glGenRenderbuffers(1, &depth);
	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, resolution.x, resolution.y);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

	GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, drawBuffers);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

Water::render_target::~render_target() noexcept
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	if (glIsFramebuffer(fbo))
		glDeleteFramebuffers(1, &fbo);
	if (glIsTexture(texture))
		glDeleteTextures(1, &texture);

	if (glIsTexture(depth))
		glDeleteTextures(1, &depth);
	else if (glIsRenderbuffer(depth))
		glDeleteRenderbuffers(1, &depth);
}

float Water::get_time() const
{
	const time_t t = clock::now();
	const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(t - start_time_);
	return static_cast<float>(static_cast<double>(delta.count()) / 500.0);
}

vec2i Water::reflection_resolution() const
{
	glm::vec2 size = resolution_;

	const float scale = 1.0f / (quality_ + 1);

	size *= scale;

	return vec2i(size);
}

vec2i Water::refraction_resolution() const
{
	glm::vec2 size = resolution_;

	const float scale = 1.0f / (0.5f * quality_ + 1);

	size *= scale;


	return vec2i(size);
}


