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
	void render_refractions() const;
	void render_reflexions() const;
	void draw();

private:
	struct render_target
	{
		GLuint fbo{};
		GLuint texture{};
		GLuint depth{};

		void create_ms(GLuint width = 800, GLuint height = 600);
		void create(GLuint width = 800, GLuint height = 600);
		~render_target() noexcept;
	};
	
	render_target refractions_{};
	render_target reflexions_{};

	mesh* mesh_;
	glm::mat3x4 model_to_world_{};
	GLuint program_{};


	time_t start_time_;
	[[nodiscard]] float get_time() const;
};

#endif