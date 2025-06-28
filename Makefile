BUILD_DIR=builddir

setup:
	rm -rf ${BUILD_DIR}
	meson setup ${BUILD_DIR}

compile:
	meson compile -C ${BUILD_DIR}
	cp ${BUILD_DIR}/compile_commands.json .