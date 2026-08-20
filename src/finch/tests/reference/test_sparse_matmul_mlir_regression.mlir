Compiling MLIR code:
module {
  func.func @scansearch(
    %arr: memref<?xindex>, %x: index, %lo: index, %hi: index
  ) -> index attributes {llvm.emit_c_interface} {
    %1 = arith.constant 1 : index
    %g:2 = scf.while (%d = %1, %p = %lo) : (index, index) -> (index, index) {
      %plt = arith.cmpi slt, %p, %hi : index
      %cond = scf.if %plt -> (i1) {
        %ap = memref.load %arr[%p] : memref<?xindex>
        %al = arith.cmpi slt, %ap, %x : index
        scf.yield %al : i1
      } else {
        %f = arith.constant false
        scf.yield %f : i1
      }
      scf.condition(%cond) %d, %p : index, index
    } do {
    ^bb0(%d: index, %p: index):
      %d2 = arith.shli %d, %1 : index
      %p2 = arith.addi %p, %d2 : index
      scf.yield %d2, %p2 : index, index
    }
    %lo1 = arith.subi %g#1, %g#0 : index
    %minp = arith.minsi %g#1, %hi : index
    %hi1 = arith.addi %minp, %1 : index
    %b:2 = scf.while (%l = %lo1, %h = %hi1) : (index, index) -> (index, index) {
      %hm1 = arith.subi %h, %1 : index
      %go = arith.cmpi slt, %l, %hm1 : index
      scf.condition(%go) %l, %h : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %diff = arith.subi %h, %l : index
      %half = arith.shrsi %diff, %1 : index
      %m = arith.addi %l, %half : index
      %am = memref.load %arr[%m] : memref<?xindex>
      %al = arith.cmpi slt, %am, %x : index
      %l2, %h2 = scf.if %al -> (index, index) {
        scf.yield %m, %h : index, index
      } else {
        scf.yield %l, %m : index, index
      }
      scf.yield %l2, %h2 : index, index
    }
    return %b#1 : index
  }

  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_15[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_16[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_15 = llvm.extractvalue %_A_16[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = builtin.unrealized_conversion_cast %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_17 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = builtin.unrealized_conversion_cast %v_21 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_23 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = arith.constant 0 : index
    %v_28 = memref.dim %v_22, %v_27 : memref<?xf64>
    %v_29 = arith.constant 1 : index
    scf.for %v_30 = %v_27 to %v_28 step %v_29 {
      %v_31 = arith.constant 0.0 : f64
      memref.store %v_31, %v_22[%v_30] : memref<?xf64>
    }
    scf.for %v_32 = %v_27 to %v_8 step %v_29 {
      %v_33 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_34 = arith.index_cast %v_33 : i64 to index
      %v_35 = arith.muli %v_34, %v_32 : index
      %v_36 = llvm.extractvalue %_A_15[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_37 = arith.index_cast %v_36 : i64 to index
      %v_38 = arith.muli %v_37, %v_32 : index
      %v_39 = memref.load %v_4[%v_38] : memref<?xindex>
      %v_40 = arith.addi %v_29, %v_38 : index
      %v_41 = memref.load %v_4[%v_40] : memref<?xindex>
      %v_42 = arith.cmpi slt, %v_39, %v_41 : index
      %v_46, %v_47 = scf.if %v_42 -> (index, index) {
        %v_43 = memref.load %v_6[%v_39] : memref<?xindex>
        %v_44 = arith.subi %v_41, %v_29 : index
        %v_45 = memref.load %v_6[%v_44] : memref<?xindex>
        scf.yield %v_45, %v_43 : index, index
      } else {
        scf.yield %v_27, %v_29 : index, index
      }
      %v_48 = memref.load %v_6[%v_39] : memref<?xindex>
      %v_49 = arith.cmpi slt, %v_48, %v_27 : index
      %v_52 = scf.if %v_49 -> (index) {
        %v_50 = arith.subi %v_41, %v_29 : index
        %v_51 = func.call @scansearch(%v_6, %v_27, %v_39, %v_50) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_51 : index
      } else {
        scf.yield %v_39 : index
      }
      %v_130:3 = scf.while (%v_53 = %v_27, %v_54 = %v_52, %v_55 = %v_47) : (index, index, index) -> (index, index, index) {
        %v_56 = arith.addi %v_29, %v_46 : index
        %v_57 = arith.minsi %v_10, %v_56 : index
        %v_58 = arith.cmpi slt, %v_55, %v_57 : index
        scf.condition(%v_58) %v_53, %v_54, %v_55 : index, index, index
      } do {
        ^bb_2(%v_53: index, %v_54: index, %v_55: index):
        scf.for %v_59 = %v_27 to %v_20 step %v_29 {
          %v_60 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_61 = arith.index_cast %v_60 : i64 to index
          %v_62 = arith.muli %v_61, %v_59 : index
          %v_63 = arith.addi %v_35, %v_62 : index
          %v_64 = memref.load %v_22[%v_63] : memref<?xf64>
          memref.store %v_64, %v_22[%v_63] : memref<?xf64>
        }
        %v_65 = arith.maxsi %v_53, %v_55 : index
        %v_66 = arith.addi %v_29, %v_55 : index
        scf.for %v_67 = %v_65 to %v_66 step %v_29 {
          %v_68 = llvm.extractvalue %_A_16[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_69 = arith.index_cast %v_68 : i64 to index
          %v_70 = arith.muli %v_69, %v_67 : index
          %v_71 = memref.load %v_14[%v_70] : memref<?xindex>
          %v_72 = arith.addi %v_29, %v_70 : index
          %v_73 = memref.load %v_14[%v_72] : memref<?xindex>
          %v_74 = arith.cmpi slt, %v_71, %v_73 : index
          %v_78, %v_79 = scf.if %v_74 -> (index, index) {
            %v_75 = memref.load %v_16[%v_71] : memref<?xindex>
            %v_76 = arith.subi %v_73, %v_29 : index
            %v_77 = memref.load %v_16[%v_76] : memref<?xindex>
            scf.yield %v_77, %v_75 : index, index
          } else {
            scf.yield %v_27, %v_29 : index, index
          }
          %v_80 = memref.load %v_16[%v_71] : memref<?xindex>
          %v_81 = arith.cmpi slt, %v_80, %v_27 : index
          %v_84 = scf.if %v_81 -> (index) {
            %v_82 = arith.subi %v_73, %v_29 : index
            %v_83 = func.call @scansearch(%v_16, %v_27, %v_71, %v_82) : (memref<?xindex>, index, index, index) -> index
            scf.yield %v_83 : index
          } else {
            scf.yield %v_71 : index
          }
          %v_116:3 = scf.while (%v_85 = %v_27, %v_86 = %v_84, %v_87 = %v_79) : (index, index, index) -> (index, index, index) {
            %v_88 = arith.addi %v_29, %v_78 : index
            %v_89 = arith.minsi %v_20, %v_88 : index
            %v_90 = arith.cmpi slt, %v_87, %v_89 : index
            scf.condition(%v_90) %v_85, %v_86, %v_87 : index, index, index
          } do {
            ^bb(%v_85: index, %v_86: index, %v_87: index):
            %v_91 = arith.addi %v_29, %v_87 : index
            %v_92 = arith.minsi %v_91, %v_87 : index
            scf.for %v_93 = %v_85 to %v_92 step %v_29 {
              %v_94 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
              %v_95 = arith.index_cast %v_94 : i64 to index
              %v_96 = arith.muli %v_95, %v_93 : index
              %v_97 = arith.addi %v_35, %v_96 : index
              %v_98 = memref.load %v_22[%v_97] : memref<?xf64>
              memref.store %v_98, %v_22[%v_97] : memref<?xf64>
            }
            %v_99 = arith.maxsi %v_85, %v_87 : index
            %v_100 = arith.addi %v_29, %v_87 : index
            scf.for %v_101 = %v_99 to %v_100 step %v_29 {
              %v_102 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
              %v_103 = arith.index_cast %v_102 : i64 to index
              %v_104 = arith.muli %v_103, %v_101 : index
              %v_105 = arith.addi %v_35, %v_104 : index
              %v_106 = memref.load %v_22[%v_105] : memref<?xf64>
              %v_107 = memref.load %v_2[%v_54] : memref<?xf64>
              %v_108 = memref.load %v_12[%v_86] : memref<?xf64>
              %v_109 = arith.mulf %v_107, %v_108 : f64
              %v_110 = arith.addf %v_106, %v_109 : f64
              memref.store %v_110, %v_22[%v_105] : memref<?xf64>
            }
            %v_111 = arith.addi %v_29, %v_87 : index
            %v_112 = arith.addi %v_29, %v_86 : index
            %v_113 = arith.cmpi slt, %v_112, %v_73 : index
            %v_115 = scf.if %v_113 -> (index) {
              %v_114 = memref.load %v_16[%v_112] : memref<?xindex>
              scf.yield %v_114 : index
            } else {
              scf.yield %v_20 : index
            }
            scf.yield %v_111, %v_112, %v_115 : index, index, index
          }
          %v_117 = arith.addi %v_29, %v_78 : index
          %v_118 = arith.maxsi %v_27, %v_117 : index
          scf.for %v_119 = %v_118 to %v_20 step %v_29 {
            %v_120 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_121 = arith.index_cast %v_120 : i64 to index
            %v_122 = arith.muli %v_121, %v_119 : index
            %v_123 = arith.addi %v_35, %v_122 : index
            %v_124 = memref.load %v_22[%v_123] : memref<?xf64>
            memref.store %v_124, %v_22[%v_123] : memref<?xf64>
          }
        }
        %v_125 = arith.addi %v_29, %v_55 : index
        %v_126 = arith.addi %v_29, %v_54 : index
        %v_127 = arith.cmpi slt, %v_126, %v_41 : index
        %v_129 = scf.if %v_127 -> (index) {
          %v_128 = memref.load %v_6[%v_126] : memref<?xindex>
          scf.yield %v_128 : index
        } else {
          scf.yield %v_10 : index
        }
        scf.yield %v_125, %v_126, %v_129 : index, index, index
      }
      scf.for %v_131 = %v_27 to %v_20 step %v_29 {
        %v_132 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_133 = arith.index_cast %v_132 : i64 to index
        %v_134 = arith.muli %v_133, %v_131 : index
        %v_135 = arith.addi %v_35, %v_134 : index
        %v_136 = memref.load %v_22[%v_135] : memref<?xf64>
        memref.store %v_136, %v_22[%v_135] : memref<?xf64>
      }
    }
    %v_137 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_138 = llvm.insertvalue %_A_7, %v_137[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_138, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
