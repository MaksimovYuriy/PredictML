# Isolated original Ruby services: replace only DB/file/calendar boundaries.
require 'json'
require 'ostruct'
require_relative 'material_forecast_service'
require_relative 'price_analysis_service'
class Service
  class Collection < Array
    def includes(**); self; end
  end
  def self.all; @all; end
  def self.load(rows)
    @all = Collection.new(rows.map do |r|
      s = OpenStruct.new(r)
      s.define_singleton_method(:active?) { active }
      s
    end)
  end
end
input = JSON.parse(STDIN.read)
Service.load(input['services'])
months = input['months']
bookings = {}
notes = {}
firsts = {}
input['counts'].each do |sid, year, month, count, revenue, avg, first|
  [bookings, notes, firsts].each { |h| h[sid] ||= {}; h[sid][year] ||= {} }
  bookings[sid][year][month] = count
  notes[sid][year][month] = { count: count, sum: revenue, avg: avg }
  firsts[sid][year][month] = first
end
material = MaterialForecastService.new
material.define_singleton_method(:generate_months) { months }
material.define_singleton_method(:preload_bookings) { bookings }
material.define_singleton_method(:export_materials_json) {}
material.define_singleton_method(:export_service_materials_json) {}
material.define_singleton_method(:write_csv) { |rows| @rows = rows }
material.call
price = PriceAnalysisService.new
price.define_singleton_method(:generate_months) { |_| months }
price.define_singleton_method(:preload_notes_data) { notes }
price.define_singleton_method(:preload_first_visit_counts) { firsts }
price.define_singleton_method(:compute_material_costs) { |_| input['costs'].transform_keys(&:to_i) }
price.define_singleton_method(:write_csv) { |rows| @rows = rows }
price.call
puts JSON.generate({ materials: material.instance_variable_get(:@rows), price: price.instance_variable_get(:@rows) })
