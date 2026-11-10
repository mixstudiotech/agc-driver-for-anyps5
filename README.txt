AnyPS5 的原生 GPU 驱动：用 Rust 实现 PS5 的 Agc 库，录制时直接写主机 GPU 的命令。
- runtime：Agc 前端、AMD 目标（RDNA3.5、RDNA4）、WDDM 后端、guest 内存导入、呈现。
- shader：把 guest gfx1013 着色器翻译成主机 ISA。
- pm4-proc、isa-proc：由 Mesa 和 AMD 的定义生成寄存器、PM4 和 ISA。
现状、构建方法见 README.md，决策见 decisions.md，任务和进度见 PLAN.md。
