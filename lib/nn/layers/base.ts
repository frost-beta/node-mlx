import {core as mx} from '../../core';
import {
  NestedDict,
  deepEqual,
  isDict,
  toCamelCase,
  toSnakeCase,
  treeFlatten,
  treeUnflatten,
} from '../../utils';

/**
 * Base class for building neural networks with MLX.
 *
 * @remarks
 *
 * All the layers provided in `mlx.nn.layers` subclass this class and your
 * models should do the same.
 *
 * A `Module` can contain other `Module` instances or `mlx.core.array` instances
 * in arbitrary nesting of python lists or dicts. The `Module` then allows
 * recursively extracting all the `mlx.core.array` instances using
 * `mlx.nn.Module.parameters`.
 *
 * In addition, the `Module` has the concept of trainable and non-trainable
 * parameters (called "frozen"). When using `mlx.nn.valueAndGrad`, the gradients
 * are returned only with respect to the trainable parameters. All arrays in a
 * module are trainable unless they are added in the "frozen" set by calling
 * `freeze`.
 *
 * @example
 * ```typescript
 * import {core as mx, nn} from '@frost-beta/mlx';
 * import {Module, Linear} from nn;
 *
 * class MyMLP extends Module {
 *   inProj: Linear;
 *   outProj: Linear;
 *
 *   constructor(inDims: number, outDims: number, hiddenDims = 16) {
 *     super();
 *     this.inProj = new Linear(inDims, hiddenDims);
 *     this.outProj = new Linear(hiddenDims, outDims);
 *   }
 *
 *   forward(x: mx.array): mx.array {
 *     let y = this.inProj.forward(x);
 *     y = mx.maximum(y, 0);
 *     return this.outProj.forward(y);
 *   }
 * }
 *
 * const model = new MyMLP(2, 1);
 *
 * // All the model parameters are created but since MLX is lazy by
 * // default, they are not evaluated yet. Calling `mx.eval` actually
 * // allocates memory and initializes the parameters.
 * mx.eval(model.parameters());
 *
 * // Setting a parameter to a new value is as simple as accessing that
 * // parameter and assigning a new array to it.
 * model.inProj.weight = mx.multiply(model.inProj.weight, 2);
 * mx.eval(model.parameters());
 * ```
 */
export abstract class Module {
  static isModule(value: unknown): value is Module {
    return value instanceof Module;
  }

  static validChildFilter(m: Module, k: string, v: unknown): boolean {
    return Array.isArray(v) || (typeof v === 'object' && isDict(v));
  }

  static validParameterFilter(m: Module, k: string, v: unknown): boolean {
    if (k.toString().startsWith('_'))
      return false;
    if (typeof v !== 'object')
      return false;
    return Array.isArray(v) || isDict(v) || (v instanceof mx.array);
  }

  static trainableParameterFilter(m: Module, k: string, v: unknown): boolean {
    return Module.validParameterFilter(m, k, v) && !m.#noGrad.has(k);
  }

  // Private members.
  #noGrad = new Set();
  #training = true;

  // Allow assigning arbitrary properties to the module.
  [key: string]: unknown;

  /**
   * Boolean indicating if the model is in training mode.
   *
   * @defaultValue `true`
   */
  get training(): boolean {
    return this.#training;
  }

  /**
   * The module's state dictionary.
   *
   * @remarks
   *
   * The module's state dictionary contains any attribute set on the module
   * including parameters in `Module.parameters`.
   *
   * Unlike `Module.parameters`, the `Module.state` property is a reference to
   * the module's state. Updates to it will be reflected in the original module.
   */
  get state(): this {
    return this;
  }

  // Convert to string representation.
  toString(): string {
    let value = `${this.constructor.name}(${this.toStringExtra()}`;
    const children = treeFlatten(this.children(), '', Module.isModule);
    for (const [k, v] of children) {
      value += `\n  ${k}: ${v}`;
    }
    if (children.length) {
      value += '\n';
    }
    value += ')';
    return value;
  }

  // Overridden by subclasses to provide more information in string
  // representation.
  protected toStringExtra(): string {
    return '';
  }

  // Human-readable console output.
  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }

  // Emulate the items() method of Python.
  items(): [string, unknown][] {
    const results: Record<string, unknown> = {};
    for (const name of Object.getOwnPropertyNames(this)) {
      if (typeof this[name] !== 'function' && !(name in Module.prototype))
        results[name] = this[name];
    }
    return Object.entries(results);
  }

  /**
   * Define the computation performed at every call. Should be overridden by
   * all subclasses.
   */
  abstract forward(...inputs: unknown[]): unknown;

  /**
   * Update the model's weights from a `.npz` or `.safetensors` file, or a list.
   *
   * @param fileOrWeights - The path to the weights `.npz` file (`.npz` or `.safetensors`) or a list of
   * pairs of parameter names and arrays.
   * @param strict - If `true` then checks that the provided weights exactly match the parameters of the
   * model. Otherwise, only the weights actually contained in the model are loaded and shapes are not checked.
   * Default: `true`.
   *
   * @returns The module instance after updating the weights.
   *
   * @example
   * ```typescript
   * import {core as mx, nn} from '@frost-beta/mlx';
   *
   * let model = new nn.Linear(10, 10);
   *
   * // Load from file
   * model.loadWeights('weights.npz');
   *
   * // Load from .safetensors file
   * model.loadWeights('weights.safetensors');
   *
   * // Load from list
   * let weights = [
   *   ['weight', mx.random.uniform(0, 1, [10, 10])],
   *   ['bias', mx.zeros([10])],
   * ];
   * model.loadWeights(weights);
   *
   * // Missing weight raise exception
   * weights = [
   *   ['weight', mx.random.uniform(0, 1, [10, 10])]
   * ];
   * try {
   *   model.loadWeights(weights);
   * } catch (e) {
   *   console.log(e);
   * }
   *
   * // Ok, only updates the weight but not the bias
   * model.loadWeights(weights, false);
   * ```
   */
  loadWeights(fileOrWeights: string | [string, mx.array][], strict = true): this {
    let weights = fileOrWeights;
    if (typeof weights === 'string') {
      weights = Object.entries(mx.load(weights));
    }

    if (strict) {
      const params = treeFlatten(this.parameters(), '', null, toSnakeCase);
      const newWeights = Object.fromEntries(weights);
      const currentWeights = Object.fromEntries(params) as Record<string, mx.array>;
      const extras = Object.keys(newWeights).filter(key => !(key in currentWeights));
      if (extras.length > 0) {
        throw Error(`Received parameters not in model: ${extras.join(' ')}.`);
      }
      const missing = Object.keys(currentWeights).filter(key => !(key in newWeights));
      if (missing.length > 0) {
        throw Error(`Missing parameters: ${missing.join(' ')}.`);
      }

      Object.keys(currentWeights).forEach(key => {
        const vNew = newWeights[key];
        if (!(vNew instanceof mx.array)) {
          throw Error(`Expected mx.array but received ${typeof vNew} for parameter ${key}`);
        }
        if (!deepEqual(vNew.shape, currentWeights[key].shape)) {
          throw Error(`Expected shape ${currentWeights[key].shape} but received shape ${vNew.shape} for parameter ${key}`);
        }
      });
    }

    if (weights.length > 0)
      this.update(treeUnflatten(weights, toCamelCase) as NestedDict<mx.array>);
    return this;
  }

  /**
   * Save the model's weights to a file.
   *
   * @remarks
   *
   * The saving method is determined by the file extension:
   * - `.npz` will use `mx.savez`
   * - `.safetensors` will use `mx.saveSafetensors`
   *
   * @param filepath - The name of the file to save the weights to.
   */
  saveWeights(filepath: string): void {
    const params = treeFlatten(this.parameters(), '', null, toSnakeCase);
    const weights = Object.fromEntries(params) as Record<string, mx.array>;
    if (filepath.endsWith('.npz')) {
      throw Error('Support for .npz format has not been implemented yet.');
    } else if (filepath.endsWith('.safetensors')) {
      mx.saveSafetensors(filepath, weights);
    } else {
      throw Error("Unsupported file extension. Use '.npz' or '.safetensors'.");
    }
  }

  /**
   * Recursively filter the contents of the module using `filterFn`, namely only
   * select keys and values where `filterFn` returns true.
   *
   * @remarks
   *
   * This is used to implement `parameters` and `trainableParameters`, but it
   * can also be used to extract any subset of the module's parameters.
   *
   * @param filterFn - Given a value, the key in which it is found, and the
   * containing module, decide whether to keep the value or drop it.
   * @param mapFn - Optionally transform the value before returning it.
   * (default: identity function).
   * @param isLeafFn - Given a value, the key in which it is found, and the
   * containing module, decide if it is a leaf (default: `true` if the value is
   * not a `Module`, `dict`, or `list`).
   *
   * @returns A dictionary containing the contents of the module recursively
   * filtered.
   */
  filterAndMap<U>(filterFn: (m: Module, k: string, v: unknown) => boolean,
                  mapFn: ((x: unknown) => U) = x => x as U,
                  isLeafFn: ((m: Module, k: string, v: unknown) => boolean) = defaultIsLeafFn): Record<string, U> {
    const result: Record<string, U> = {};
    for (let [k, v] of this.items()) {
      if (filterFn(this, k, v))
        result[k] = unwrap(this, k, v, filterFn, mapFn, isLeafFn) as U;
    }
    return result;
  }

  /**
   * Recursively return all the `mlx.core.array` members of this Module as a
   * dict of dicts and lists.
   */
  parameters(): NestedDict<mx.array> {
    return this.filterAndMap(Module.validParameterFilter) as NestedDict<mx.array>;
  }

  /**
   * Recursively return all the non-frozen `mlx.core.array` members of this
   * Module as a dict of dicts and lists.
   */
  trainableParameters(): NestedDict<mx.array> {
    return this.filterAndMap(Module.trainableParameterFilter) as NestedDict<mx.array>;
  }

  /**
   * Return the direct descendants of this Module instance.
   */
  children(): NestedDict<Module> {
    const isLeaf = (m: Module, k: string, v: unknown) => v instanceof Module;
    return this.filterAndMap(Module.validChildFilter, undefined, isLeaf) as NestedDict<Module>;
  }

  /**
   * Return the submodules that do not contain other modules.
   */
  leafModules(): NestedDict<Module> {
    const isLeafModule = (m: this, k: string, v: unknown) => {
      return v instanceof Module && treeFlatten(v.children()).length === 0;
    };
    return this.filterAndMap(Module.validChildFilter, undefined, isLeafModule) as NestedDict<Module>;
  }

  /**
   * Replace the parameters of this Module with the provided ones in the dict of
   * dicts and lists.
   *
   * @remarks
   *
   * Commonly used by the optimizer to change the model to the updated
   * (optimized) parameters. Also used by the `mlx.nn.valueAndGrad` to set the
   * tracers in the model in order to compute gradients.
   *
   * The passed-in parameters dictionary need not be a full dictionary similar
   * to `parameters`. Only the provided locations will be updated.
   *
   * @param parameters - A complete or partial dictionary of the module's
   * parameters.
   *
   * @returns The module instance after updating the parameters.
   */
  update(parameters: NestedDict<mx.array>): this {
    const apply = (target: Record<string, any>, parameters: NestedDict<mx.array>) => {
      if (typeof parameters !== 'object')
        return;
      for (let k in parameters) {
        if (!(k in target))
          continue;
        const value = target[k];
        const newValue = parameters[k] as NestedDict<mx.array>;
        if (value instanceof mx.array)
          target[k] = newValue;
        else if (value instanceof Module)
          value.update(newValue);
        else if (typeof value === 'object')
          apply(value, newValue);
      }
    };
    apply(this, parameters);
    return this;
  }

  /**
   * Map all the parameters using the provided `mapFn` and immediately update
   * the module with the mapped parameters.
   *
   * @remarks
   *
   * For instance running `model.apply(x => x.astype(mx.float16))` casts all
   * parameters to 16 bit floats.
   *
   * @param mapFn - Maps an array to another array.
   * @param filterFn - Filter to select which arrays to map. Default:
   * `Module.validParameterFilter`.
   *
   * @returns The module instance after updating the parameters.
   */
  apply(mapFn: (param: mx.array) => mx.array,
        filterFn: (m: Module, k: string, v: unknown) => boolean = Module.validParameterFilter): this {
    return this.update(this.filterAndMap(filterFn, mapFn));
  }

  /**
   * Replace the child modules of this `Module` instance with the provided ones
   * in the dictionary of dictionaries and lists.
   *
   * @remarks
   *
   * It is the equivalent of `Module.update` but for modules instead of
   * parameters and allows us to flexibly edit complex architectures by
   * programmatically swapping layers.
   *
   * The passed in parameters dictionary need not be a full dictionary similar
   * to `parameters()`. Only the provided locations will be updated.
   *
   * @param modules - A complete or partial dictionary of the modules
   * submodules.
   *
   * @returns The module instance after updating the submodules.
   */
  updateModules(modules: NestedDict<Module>): this {
    const apply = (target: Record<string, any>, modules: Record<string, any>) => {
      if (typeof modules !== 'object')
        return;
      for (let k in modules) {
        if (!(k in target))
          continue;
        const value = target[k];
        const newValue = modules[k];
        if (Module.isModule(value) && Module.isModule(newValue))
          target[k] = newValue;
        else if (typeof value === 'object' && typeof newValue === 'object')
          apply(value, newValue);
      }
    };
    apply(this, modules);
    return this;
  }

  /**
   * Apply a function to all the modules in this instance (including this
   * instance).
   *
   * @param applyFn - The function to apply to the modules.
   *
   * @returns The module instance after updating submodules.
   */
  applyToModules(applyFn: (prefix: string, m: Module) => unknown): this {
    const stack: [string, Module][] = [['', this]];
    while (stack.length > 0) {
      let [prefix, m] = stack.pop();
      applyFn(prefix, m);
      prefix = prefix ? `.${prefix}` : '';
      stack.push(...treeFlatten(m.children(), prefix, Module.isModule) as [string, Module][]);
    }
    return this;
  }

  /**
   * Return a list with all the modules in this instance.
   *
   * @returns An array of `Module` instances.
   */
  modules(): Module[] {
    const moduleList: Module[] = [];
    this.applyToModules((prefix: string, m: Module) => moduleList.push(m));
    return moduleList;
  }

  /**
   * Return a list with all the modules in this instance and their name with dot
   * notation.
   *
   * @returns An array of tuples (string, `Module`).
   */
  namedModules(): [string, Module][] {
    const moduleList: [string, Module][] = [];
    this.applyToModules((prefix: string, m: Module) => moduleList.push([prefix, m]));
    return moduleList;
  }

  /**
   * Freeze the Module's parameters or some of them.
   *
   * @remarks
   *
   * Freezing a parameter means not computing gradients for it. This function is
   * idempotent i.e. freezing a frozen model is a no-op.
   *
   * @param recurse - If `true` then freeze the parameters of the submodules as
   * well. Default: `true`.
   * @param keys - If provided then only these parameters will be frozen
   * otherwise all the parameters of a module. For instance freeze all biases by
   * calling `model.freeze(true, 'bias')`.
   * @param strict - If set to `true` validate that the passed keys exist.
   * Default: `false`.
   *
   * @returns The module instance after freezing the parameters.
   *
   * @example
   * For instance to only train the attention parameters from a Transformer:
   *
   * ```typescript
   * const model = new Transformer();
   * model.freeze();
   * model.applyToModules((k: string, v: Module) => { if (k.endsWith('attention')) v.unfreeze() });
   * ```
   */
  freeze(recurse = true, keys?: string | string[], strict = false): this {
    const freezeImpl = (prefix: string, m: Module) => {
      let localKeys = keys;
      if (keys == null) {
        const params = m.filterAndMap((m, k, v) => !(v instanceof Module) && Module.validParameterFilter(m, k, v));
        localKeys = treeFlatten(params).map(([k, v]) => k);
      }
      m.validateKeys(localKeys, strict)
       .forEach(k => m.#noGrad.add(k));
    };

    if (recurse) {
      this.applyToModules(freezeImpl);
    } else {
      freezeImpl('', this);
    }
    return this;
  }

  /**
   * Unfreeze the Module's parameters or some of them.
   *
   * @remarks
   *
   * This function is idempotent ie unfreezing a model that is not frozen is a
   * noop.
   *
   * @param recurse - If true then unfreeze the parameters of the submodules as
   * well. Default: `true`.
   * @param keys - If provided then only these parameters will be unfrozen
   * otherwise all the parameters of a module. For instance unfreeze all biases
   * by calling `module.unfreeze(true, 'bias')`.
   * @param strict - If set to `true` validate that the passed keys exist.
   * Default: `false`.
   *
   * @returns The module instance after unfreezing the parameters.
   *
   * @example
   * For instance to only train the biases of a Transformer one can do:
   *
   * ```typescript
   * import {nn} from '@frost-beta/mlx';
   * const model = new nn.Transformer();
   * model.freeze();
   * model.unfreeze(true, 'bias');
   * ```
   */
  unfreeze(recurse = true, keys?: string | string[], strict = false): this {
    const unfreezeImpl = (prefix: string, m: Module) => {
      if (keys == null) {
        m.#noGrad.clear();
      } else {
        m.validateKeys(keys, strict)
         .forEach(k => m.#noGrad.delete(k));
      }
    };

    if (recurse) {
      this.applyToModules(unfreezeImpl);
    } else {
      unfreezeImpl('', this);
    }
    return this;
  }

  /**
   * Set the model in or out of training mode.
   *
   * @remarks
   *
   * Training mode only applies to certain layers. For example `Dropout` applies
   * a random mask in training mode, but is the identity in evaluation mode.
   *
   * @param mode - Indicate if the model should be in training or evaluation
   * mode. Default: `true`.
   *
   * @returns The module instance after updating the training mode.
   */
  train(mode = true): this {
    return this.applyToModules((prefix, m) => m.#training = mode);
  }

  /**
   * Set the model to evaluation mode.
   */
  eval(): this {
    return this.train(false);
  }

  /**
   * Set the dtype of the module's parameters.
   *
   * @param dtype - The new dtype.
   * @param predicate - A predicate to select parameters to cast. By default,
   * only parameters of type `mx.floating` will be updated to avoid casting
   * integer parameters to the new dtype.
   */
  setDtype(dtype: mx.Dtype,
           predicate: ((x: mx.Dtype) => boolean) | null = (x: mx.Dtype) => mx.issubdtype(x, mx.floating)): void {
    this.apply((x: mx.array) => (!predicate || predicate(x.dtype)) ? x.astype(dtype) : x);
  }

  private validateKeys(keys: string | string[], strict: boolean): string[] {
    keys = typeof keys === 'string' ? [keys] : keys;
    if (strict) {
      for (const k of keys) {
        if (!(k in this))
          throw Error(`Module doesn't contain member ${k}.`);
      }
    }
    return keys;
  }
}

// Helpers.
function defaultIsLeafFn(m: Module, k: string, v: unknown) {
  if (typeof v !== 'object')
    return true;
  return !Array.isArray(v) && !isDict(v);
}

function unwrap(model: Module,
                valueKey: string,
                value: unknown,
                filterFn: (m: Module, k: string, v: unknown) => boolean,
                mapFn: (value: unknown) => unknown,
                isLeafFn: (m: Module, k: string, v: unknown) => boolean): unknown {
  if (isLeafFn(model, valueKey, value)) {
    return mapFn(value);
  }

  if (Module.isModule(value)) {
    const newValue: Record<string, unknown> = {};
    for (const [k, v] of value.items()) {
      if (filterFn(value, k, v))
        newValue[k] = unwrap(value, k, v, filterFn, mapFn, isLeafFn);
    }
    return newValue;
  }

  if (Array.isArray(value)) {
    const newValue: unknown[] = [];
    for (let i = 0; i < value.length; i++) {
      let key = `${valueKey}.${i}`;
      if (filterFn(model, key, value[i])) {
        newValue.push(unwrap(model, key, value[i], filterFn, mapFn, isLeafFn));
      } else {
        newValue.push({});
      }
    }
    return newValue;
  }

  if (typeof value === 'object') {
    const newValue: Record<string, unknown> = {};
    for (let k in value) {
      const key = `${valueKey}.${k}`;
      const v = value[k as keyof typeof value];
      if (filterFn(model, key, v)) {
        newValue[k] = unwrap(model, key, v, filterFn, mapFn, isLeafFn);
      } else {
        newValue[k] = {};
      }
    }
    return newValue;
  }

  throw Error("Unexpected leaf found while traversing the module");
}
