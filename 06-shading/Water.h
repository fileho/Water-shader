#ifndef WATER_H
#define WATER_H

#include "Mesh.h"
#include "Geometry.h"
#include "glm/glm.hpp"
#include <glad/glad.h>
#include <chrono>

class Water
{
	typedef Mesh<Vertex_Pos_Tex> mesh;
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::time_point<std::chrono::steady_clock> time_t;
	
public:
	explicit Water();
	~Water() noexcept;

	void set_program(GLuint program);
	void set_bump_map(GLuint bump_map) { bump_map_ = bump_map; }
	void render_refractions() const;
	void render_reflexions() const;
	void draw(const glm::vec4& viewPosWS);
	void set_resolution(int width, int height);

private:
	struct render_target
	{
		GLuint fbo{};
		GLuint texture{};
		GLuint depth{};

		void create_depth(glm::vec<2, GLuint> resolution);
		void create(glm::vec<2, GLuint> resolution);
		
		~render_target() noexcept;
	};

	render_target refractions_{};
	render_target reflections_{};

	mesh *mesh_;
	glm::mat3x4 model_to_world_{};
	GLuint program_{};
	GLuint bump_map_{};


	time_t start_time_;
	glm::vec2 resolution_ = glm::vec2(800, 600);

	const float tau_ = 6.28318530718f;
	[[nodiscard]] float get_time() const;
};

#endif