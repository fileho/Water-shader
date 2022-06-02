#ifndef WATER_H
#define WATER_H

#include "Mesh.h"
#include "Geometry.h"
#include "glm/glm.hpp"
#include <glad/glad.h>
#include <chrono>

typedef glm::vec<2, GLuint> vec2i;

class Water
{
	typedef Mesh<Vertex_Pos> mesh;
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::time_point<std::chrono::steady_clock> time_t;
	
public:
	explicit Water();
	~Water() noexcept;

	void set_program(GLuint program);
	void set_bump_map(GLuint bump_map) { bump_map_ = bump_map; }
	void render_refractions() const;
	void render_reflections() const;
	void draw(const glm::vec4& viewPosWS);
	void set_resolution(int width, int height);
	void set_quality(float lod);
private:
	struct render_target
	{
		GLuint fbo{};
		GLuint texture{};
		GLuint depth{};

		void create_depth(vec2i resolution);
		void create(vec2i resolution);
		
		~render_target() noexcept;
	};

	render_target refractions_{};
	render_target reflections_{};

	mesh *mesh_;
	glm::mat3x4 model_to_world_{};
	GLuint program_{};
	GLuint bump_map_{};
	float quality_{};

	time_t start_time_;
	glm::vec<2, GLuint> resolution_ = glm::vec2(800, 600);
	const float tau_ = 6.28318530718f;
	[[nodiscard]] float get_time() const;
	[[nodiscard]] vec2i reflection_resolution() const;
	[[nodiscard]] vec2i refraction_resolution() const;
};

#endif